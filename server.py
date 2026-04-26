from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union
import subprocess
import json
import shutil
import os
import statistics
import cv2

app = FastAPI()


# =========================
# Models
# =========================

class VideoInfoRequest(BaseModel):
    input_path: str


class AnalyzeCropRequest(BaseModel):
    input_path: str
    start: Optional[Union[str, float, int]] = None
    end: Optional[Union[str, float, int]] = None
    output_width: int = 720
    output_height: int = 1280
    samples: int = 12


class ProcessAutoCropRequest(BaseModel):
    input_path: str
    output_path: str
    start: Optional[Union[str, float, int]] = None
    end: Optional[Union[str, float, int]] = None
    output_width: int = 720
    output_height: int = 1280
    samples: int = 12
    subtitle_path: Optional[str] = None
    crf: int = Field(default=20, ge=16, le=30)


class ExtractAudioRequest(BaseModel):
    input_path: str
    output_path: str
    start: Optional[Union[str, float, int]] = None
    end: Optional[Union[str, float, int]] = None


class CleanupRequest(BaseModel):
    job_path: str


# =========================
# Helpers
# =========================

def run_command(command, timeout=3600):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Command failed",
                "command": command,
                "stderr": result.stderr[-4000:]
            }
        )

    return result


def ensure_file_exists(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo não encontrado: {path}"
        )


def ensure_parent_folder(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_time(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if ":" not in value:
        return float(value)

    parts = value.split(":")
    parts = [float(p) for p in parts]

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    raise ValueError(f"Formato de tempo inválido: {value}")


def make_even(number):
    number = int(round(number))
    return number if number % 2 == 0 else number - 1


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_video_info(input_path: str):
    ensure_file_exists(input_path)

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        input_path
    ]

    result = run_command(command, timeout=60)
    data = json.loads(result.stdout)

    streams = data.get("streams", [])
    if not streams:
        raise HTTPException(status_code=400, detail="Nenhum stream de vídeo encontrado.")

    stream = streams[0]

    width = int(stream["width"])
    height = int(stream["height"])

    duration = stream.get("duration") or data.get("format", {}).get("duration")
    duration = float(duration) if duration else None

    fps = stream.get("r_frame_rate")

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps
    }


def calculate_crop(input_width, input_height, output_width, output_height, center_x=None):
    target_ratio = output_width / output_height

    crop_h = input_height
    crop_w = make_even(crop_h * target_ratio)

    if crop_w > input_width:
        crop_w = input_width
        crop_h = make_even(crop_w / target_ratio)

    if center_x is None:
        center_x = input_width / 2

    crop_x = make_even(center_x - crop_w / 2)
    crop_y = 0

    crop_x = clamp(crop_x, 0, input_width - crop_w)

    return {
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y)
    }


def analyze_crop_internal(input_path: str, start=None, end=None, output_width=720, output_height=1280, samples=12):
    info = get_video_info(input_path)

    input_width = info["width"]
    input_height = info["height"]
    duration = info["duration"]

    start_sec = parse_time(start) if start is not None else 0
    end_sec = parse_time(end) if end is not None else duration

    if duration is None:
        raise HTTPException(status_code=400, detail="Não foi possível obter a duração do vídeo.")

    if end_sec is None or end_sec > duration:
        end_sec = duration

    if start_sec >= end_sec:
        raise HTTPException(status_code=400, detail="O tempo inicial precisa ser menor que o tempo final.")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Não foi possível abrir o vídeo com OpenCV.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise HTTPException(status_code=500, detail="Não foi possível carregar o detector de rosto.")

    face_centers = []

    samples = max(3, min(samples, 30))

    if samples == 1:
        times = [(start_sec + end_sec) / 2]
    else:
        interval = (end_sec - start_sec) / (samples - 1)
        times = [start_sec + i * interval for i in range(samples)]

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        success, frame = cap.read()

        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            continue

        # Pega o maior rosto detectado
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        center_x = x + w / 2
        face_centers.append(center_x)

    cap.release()

    if face_centers:
        center_x = statistics.median(face_centers)
        detection_found = True
    else:
        center_x = input_width / 2
        detection_found = False

    crop = calculate_crop(
        input_width=input_width,
        input_height=input_height,
        output_width=output_width,
        output_height=output_height,
        center_x=center_x
    )

    return {
        "input": info,
        "output": {
            "width": output_width,
            "height": output_height
        },
        "face_detection": {
            "found": detection_found,
            "samples_used": samples,
            "faces_detected_in_frames": len(face_centers),
            "center_x": center_x
        },
        "crop": crop
    }


def escape_ass_path(path: str):
    # Para caminhos Linux simples, isso já resolve a maioria dos casos.
    return str(Path(path).resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "video-worker"
    }


@app.get("/ffmpeg-version")
def ffmpeg_version():
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True
    )

    first_line = result.stdout.splitlines()[0] if result.stdout else "FFmpeg não respondeu"

    return {
        "status": "ok" if result.returncode == 0 else "error",
        "ffmpeg": first_line
    }


@app.post("/video-info")
def video_info(request: VideoInfoRequest):
    return get_video_info(request.input_path)


@app.post("/analyze-crop")
def analyze_crop(request: AnalyzeCropRequest):
    return analyze_crop_internal(
        input_path=request.input_path,
        start=request.start,
        end=request.end,
        output_width=request.output_width,
        output_height=request.output_height,
        samples=request.samples
    )


@app.post("/process-auto-crop")
def process_auto_crop(request: ProcessAutoCropRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    analysis = analyze_crop_internal(
        input_path=request.input_path,
        start=request.start,
        end=request.end,
        output_width=request.output_width,
        output_height=request.output_height,
        samples=request.samples
    )

    crop = analysis["crop"]

    video_filter = (
        f"crop={crop['crop_w']}:{crop['crop_h']}:{crop['crop_x']}:{crop['crop_y']},"
        f"scale={request.output_width}:{request.output_height},"
        f"setsar=1"
    )

    if request.subtitle_path:
        ensure_file_exists(request.subtitle_path)
        subtitle_path = escape_ass_path(request.subtitle_path)
        video_filter += f",ass='{subtitle_path}'"

    command = ["ffmpeg", "-y"]

    start_sec = parse_time(request.start) if request.start is not None else None
    end_sec = parse_time(request.end) if request.end is not None else None

    if start_sec is not None:
        command += ["-ss", str(start_sec)]

    command += ["-i", request.input_path]

    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            if duration <= 0:
                raise HTTPException(status_code=400, detail="O tempo final precisa ser maior que o inicial.")
            command += ["-t", str(duration)]
        else:
            command += ["-to", str(end_sec)]

    command += [
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(request.crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        request.output_path
    ]

    run_command(command, timeout=7200)

    return {
        "status": "success",
        "message": "Vídeo processado com crop automático.",
        "input_path": request.input_path,
        "output_path": request.output_path,
        "analysis": analysis
    }


@app.post("/extract-audio")
def extract_audio(request: ExtractAudioRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    command = ["ffmpeg", "-y"]

    start_sec = parse_time(request.start) if request.start is not None else None
    end_sec = parse_time(request.end) if request.end is not None else None

    if start_sec is not None:
        command += ["-ss", str(start_sec)]

    command += ["-i", request.input_path]

    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            if duration <= 0:
                raise HTTPException(status_code=400, detail="O tempo final precisa ser maior que o inicial.")
            command += ["-t", str(duration)]
        else:
            command += ["-to", str(end_sec)]

    command += [
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        request.output_path
    ]

    run_command(command, timeout=3600)

    return {
        "status": "success",
        "message": "Áudio extraído.",
        "input_path": request.input_path,
        "output_path": request.output_path
    }


@app.post("/cleanup")
def cleanup(request: CleanupRequest):
    path = Path(request.job_path).resolve()

    allowed_roots = [
        Path("/tmp/video-jobs").resolve(),
        Path("/data/video-jobs").resolve(),
        Path("/data/tmp").resolve()
    ]

    is_allowed = any(str(path).startswith(str(root) + os.sep) for root in allowed_roots)

    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail="Por segurança, só é permitido apagar pastas dentro de /tmp/video-jobs, /data/video-jobs ou /data/tmp."
        )

    if not path.exists():
        return {
            "status": "ok",
            "message": "Pasta não existe. Nada para apagar.",
            "job_path": str(path)
        }

    shutil.rmtree(path)

    return {
        "status": "success",
        "message": "Pasta temporária apagada.",
        "job_path": str(path)
    }

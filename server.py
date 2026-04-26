from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union
import subprocess
import json
import shutil
import os

app = FastAPI()


# =========================
# Models
# =========================

class VideoInfoRequest(BaseModel):
    input_path: str


class AnalyzeCropRequest(BaseModel):
    input_path: str
    output_width: int = 720
    output_height: int = 1280
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    center_x: Optional[float] = None


class ProcessFixedCropRequest(BaseModel):
    input_path: str
    output_path: str
    start: Optional[Union[str, float, int]] = None
    end: Optional[Union[str, float, int]] = None
    output_width: int = 720
    output_height: int = 1280
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    center_x: Optional[float] = None
    subtitle_path: Optional[str] = None
    crf: int = Field(default=20, ge=16, le=30)


class ExtractFrameRequest(BaseModel):
    input_path: str
    output_path: str
    timestamp: Optional[Union[str, float, int]] = 5


class PreviewCropRequest(BaseModel):
    input_path: str
    output_path: str
    timestamp: Optional[Union[str, float, int]] = 5
    output_width: int = 720
    output_height: int = 1280
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    center_x: Optional[float] = None


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


def calculate_crop(
    input_width,
    input_height,
    output_width,
    output_height,
    crop_x=None,
    crop_y=None,
    center_x=None
):
    target_ratio = output_width / output_height

    crop_h = input_height
    crop_w = make_even(crop_h * target_ratio)

    if crop_w > input_width:
        crop_w = input_width
        crop_h = make_even(crop_w / target_ratio)

    if center_x is None:
        center_x = input_width / 2

    if crop_x is None:
        crop_x = center_x - crop_w / 2

    crop_x = make_even(crop_x)
    crop_x = clamp(crop_x, 0, input_width - crop_w)

    if crop_y is None:
        crop_y = 0

    crop_y = make_even(crop_y)
    crop_y = clamp(crop_y, 0, input_height - crop_h)

    return {
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y)
    }


def analyze_crop_internal(
    input_path: str,
    output_width=720,
    output_height=1280,
    crop_x=None,
    crop_y=None,
    center_x=None
):
    info = get_video_info(input_path)

    crop = calculate_crop(
        input_width=info["width"],
        input_height=info["height"],
        output_width=output_width,
        output_height=output_height,
        crop_x=crop_x,
        crop_y=crop_y,
        center_x=center_x
    )

    return {
        "input": info,
        "output": {
            "width": output_width,
            "height": output_height
        },
        "crop_mode": "fixed",
        "message": "Crop fixo/manual. Sem OpenCV, sem cv2 e sem NumPy.",
        "crop": crop
    }


def escape_ass_path(path: str):
    return str(Path(path).resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_video_filter(crop, output_width, output_height, subtitle_path=None):
    video_filter = (
        f"crop={crop['crop_w']}:{crop['crop_h']}:{crop['crop_x']}:{crop['crop_y']},"
        f"scale={output_width}:{output_height},"
        f"setsar=1"
    )

    if subtitle_path:
        ensure_file_exists(subtitle_path)
        subtitle_path = escape_ass_path(subtitle_path)
        video_filter += f",ass='{subtitle_path}'"

    return video_filter


def add_time_args(command, start=None, end=None):
    start_sec = parse_time(start) if start is not None else None
    end_sec = parse_time(end) if end is not None else None

    if start_sec is not None:
        command += ["-ss", str(start_sec)]

    return command, start_sec, end_sec


def add_duration_args(command, start_sec=None, end_sec=None):
    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            if duration <= 0:
                raise HTTPException(status_code=400, detail="O tempo final precisa ser maior que o inicial.")
            command += ["-t", str(duration)]
        else:
            command += ["-to", str(end_sec)]

    return command


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
        output_width=request.output_width,
        output_height=request.output_height,
        crop_x=request.crop_x,
        crop_y=request.crop_y,
        center_x=request.center_x
    )


@app.post("/extract-frame")
def extract_frame(request: ExtractFrameRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    timestamp = parse_time(request.timestamp)

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", request.input_path,
        "-frames:v", "1",
        "-q:v", "2",
        request.output_path
    ]

    run_command(command, timeout=300)

    return {
        "status": "success",
        "message": "Frame extraído.",
        "input_path": request.input_path,
        "output_path": request.output_path,
        "timestamp": timestamp
    }


@app.post("/preview-crop")
def preview_crop(request: PreviewCropRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    analysis = analyze_crop_internal(
        input_path=request.input_path,
        output_width=request.output_width,
        output_height=request.output_height,
        crop_x=request.crop_x,
        crop_y=request.crop_y,
        center_x=request.center_x
    )

    crop = analysis["crop"]
    video_filter = build_video_filter(
        crop=crop,
        output_width=request.output_width,
        output_height=request.output_height
    )

    timestamp = parse_time(request.timestamp)

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", request.input_path,
        "-frames:v", "1",
        "-vf", video_filter,
        "-q:v", "2",
        request.output_path
    ]

    run_command(command, timeout=300)

    return {
        "status": "success",
        "message": "Preview do crop gerado.",
        "input_path": request.input_path,
        "output_path": request.output_path,
        "timestamp": timestamp,
        "analysis": analysis
    }


@app.post("/process-fixed-crop")
def process_fixed_crop(request: ProcessFixedCropRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    analysis = analyze_crop_internal(
        input_path=request.input_path,
        output_width=request.output_width,
        output_height=request.output_height,
        crop_x=request.crop_x,
        crop_y=request.crop_y,
        center_x=request.center_x
    )

    crop = analysis["crop"]

    video_filter = build_video_filter(
        crop=crop,
        output_width=request.output_width,
        output_height=request.output_height,
        subtitle_path=request.subtitle_path
    )

    command = ["ffmpeg", "-y"]

    command, start_sec, end_sec = add_time_args(
        command,
        start=request.start,
        end=request.end
    )

    command += ["-i", request.input_path]

    command = add_duration_args(
        command,
        start_sec=start_sec,
        end_sec=end_sec
    )

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
        "message": "Vídeo processado com crop fixo.",
        "input_path": request.input_path,
        "output_path": request.output_path,
        "analysis": analysis
    }


@app.post("/process-auto-crop")
def process_auto_crop(request: ProcessFixedCropRequest):
    """
    Mantive esta rota para não quebrar o fluxo caso a gente já tenha usado o nome /process-auto-crop.
    Mas agora ela NÃO faz detecção automática por rosto.
    Ela usa crop fixo/manual com crop_x, crop_y ou center_x.
    """
    return process_fixed_crop(request)


@app.post("/extract-audio")
def extract_audio(request: ExtractAudioRequest):
    ensure_file_exists(request.input_path)
    ensure_parent_folder(request.output_path)

    command = ["ffmpeg", "-y"]

    command, start_sec, end_sec = add_time_args(
        command,
        start=request.start,
        end=request.end
    )

    command += ["-i", request.input_path]

    command = add_duration_args(
        command,
        start_sec=start_sec,
        end_sec=end_sec
    )

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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
import subprocess
import json
import shutil
import uuid
import re


app = FastAPI()


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
                "stderr": result.stderr[-5000:]
            }
        )

    return result


def safe_filename(name: str, fallback: str):
    name = name or fallback
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or fallback


async def save_upload(upload: UploadFile, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def parse_time(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    if ":" not in value:
        return float(value)

    parts = [float(p) for p in value.split(":")]

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    raise HTTPException(status_code=400, detail=f"Formato de tempo inválido: {value}")


def make_even(number):
    number = int(round(number))
    return number if number % 2 == 0 else max(0, number - 1)


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def get_media_info(input_path: str):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        input_path
    ]

    result = run_command(command, timeout=60)
    data = json.loads(result.stdout)

    streams = data.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        raise HTTPException(status_code=400, detail="Nenhum stream de vídeo encontrado.")

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    duration = (
        video_stream.get("duration")
        or data.get("format", {}).get("duration")
    )
    duration = float(duration) if duration else None

    fps = video_stream.get("r_frame_rate")

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "has_audio": audio_stream is not None
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


def escape_ass_path(path: str):
    return (
        str(Path(path).resolve())
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


def build_main_video_filter(crop, output_width, output_height, subtitle_path, fps):
    subtitle_path = escape_ass_path(subtitle_path)

    return (
        f"crop={crop['crop_w']}:{crop['crop_h']}:{crop['crop_x']}:{crop['crop_y']},"
        f"scale={output_width}:{output_height},"
        f"setsar=1,"
        f"ass='{subtitle_path}',"
        f"fps={fps},"
        f"format=yuv420p"
    )


def build_cover_video_filter(output_width, output_height, fps):
    return (
        f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
        f"crop={output_width}:{output_height},"
        f"setsar=1,"
        f"fps={fps},"
        f"format=yuv420p"
    )


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


@app.post("/render-subtitled-video")
async def render_subtitled_video(
    background_tasks: BackgroundTasks,

    # Arquivos principais
    video: UploadFile = File(...),
    legenda_ass: UploadFile = File(...),
    imagem_capa: Optional[UploadFile] = File(None),

    # Saída
    output_width: int = Form(1080),
    output_height: int = Form(1920),
    fps: int = Form(30),
    crf: int = Form(20),

    # Capa inicial
    cover_frames: int = Form(3),

    # Corte de tempo opcional
    start: Optional[str] = Form(None),
    end: Optional[str] = Form(None),

    # Crop/enquadramento opcional
    crop_x: Optional[int] = Form(None),
    crop_y: Optional[int] = Form(None),
    center_x: Optional[float] = Form(None)
):
    if crf < 16 or crf > 30:
        raise HTTPException(status_code=400, detail="CRF precisa estar entre 16 e 30.")

    if fps < 1 or fps > 120:
        raise HTTPException(status_code=400, detail="FPS precisa estar entre 1 e 120.")

    if cover_frames < 0 or cover_frames > 300:
        raise HTTPException(status_code=400, detail="cover_frames precisa estar entre 0 e 300.")

    if output_width <= 0 or output_height <= 0:
        raise HTTPException(status_code=400, detail="output_width e output_height precisam ser maiores que zero.")

    job_id = str(uuid.uuid4())
    job_dir = Path("/tmp/video-jobs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_name = safe_filename(video.filename, "input.mp4")
    ass_name = safe_filename(legenda_ass.filename, "legenda.ass")

    input_video_path = job_dir / video_name
    subtitle_path = job_dir / ass_name

    capa_path = None
    if imagem_capa is not None:
        capa_name = safe_filename(imagem_capa.filename, "capa.jpg")
        capa_path = job_dir / capa_name

    output_name = f"{Path(video_name).stem}_final.mp4"
    output_path = job_dir / output_name

    await save_upload(video, input_video_path)
    await save_upload(legenda_ass, subtitle_path)

    if imagem_capa is not None:
        await save_upload(imagem_capa, capa_path)

    info = get_media_info(str(input_video_path))

    crop = calculate_crop(
        input_width=info["width"],
        input_height=info["height"],
        output_width=output_width,
        output_height=output_height,
        crop_x=crop_x,
        crop_y=crop_y,
        center_x=center_x
    )

    start_sec = parse_time(start)
    end_sec = parse_time(end)

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="O tempo final precisa ser maior que o inicial.")

    main_video_filter = build_main_video_filter(
        crop=crop,
        output_width=output_width,
        output_height=output_height,
        subtitle_path=str(subtitle_path),
        fps=fps
    )

    has_cover = capa_path is not None and cover_frames > 0
    cover_duration = cover_frames / fps if has_cover else 0

    command = ["ffmpeg", "-y"]

    if has_cover:
        command += [
            "-loop", "1",
            "-framerate", str(fps),
            "-t", str(cover_duration),
            "-i", str(capa_path)
        ]

    # Input do vídeo principal
    if start_sec is not None:
        command += ["-ss", str(start_sec)]

    command += ["-i", str(input_video_path)]

    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            command += ["-t", str(duration)]
        else:
            command += ["-to", str(end_sec)]

    if has_cover:
        # Input de áudio silencioso para a capa inicial
        command += [
            "-f", "lavfi",
            "-t", str(cover_duration),
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"
        ]

        video_input_index = 1
        silent_audio_index = 2

        if info["has_audio"]:
            main_audio_filter = (
                f"[{video_input_index}:a]"
                f"aresample=44100,"
                f"aformat=sample_rates=44100:channel_layouts=stereo"
                f"[maina]"
            )
        else:
            # Se o vídeo original não tiver áudio, cria áudio silencioso também para o trecho principal.
            main_duration = None

            if start_sec is not None and end_sec is not None:
                main_duration = end_sec - start_sec
            elif info["duration"] is not None:
                main_duration = info["duration"] - (start_sec or 0)

            if main_duration is None or main_duration <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Não consegui determinar a duração do vídeo sem áudio."
                )

            # Neste caso, precisamos de mais um input lavfi para o áudio silencioso do vídeo principal.
            command += [
                "-f", "lavfi",
                "-t", str(main_duration),
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"
            ]

            main_silent_audio_index = 3

            main_audio_filter = (
                f"[{main_silent_audio_index}:a]"
                f"aresample=44100,"
                f"aformat=sample_rates=44100:channel_layouts=stereo"
                f"[maina]"
            )

        cover_video_filter = build_cover_video_filter(
            output_width=output_width,
            output_height=output_height,
            fps=fps
        )

        filter_complex = (
            f"[0:v]{cover_video_filter}[coverv];"
            f"[{video_input_index}:v]{main_video_filter}[mainv];"
            f"[{silent_audio_index}:a]"
            f"aresample=44100,"
            f"aformat=sample_rates=44100:channel_layouts=stereo"
            f"[covera];"
            f"{main_audio_filter};"
            f"[coverv][covera][mainv][maina]"
            f"concat=n=2:v=1:a=1[v][a]"
        )

        command += [
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]"
        ]

    else:
        # Sem capa: só renderiza o vídeo legendado.
        command += [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", main_video_filter
        ]

    command += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    run_command(command, timeout=7200)

    background_tasks.add_task(shutil.rmtree, str(job_dir), ignore_errors=True)

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_name,
        background=background_tasks
    )


@app.post("/cut-video")
async def cut_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    start: Optional[str] = Form(None),
    end: Optional[str] = Form(None),
    crf: int = Form(20)
):
    if crf < 16 or crf > 30:
        raise HTTPException(status_code=400, detail="CRF precisa estar entre 16 e 30.")

    start_sec = parse_time(start)
    end_sec = parse_time(end)

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="O tempo final precisa ser maior que o inicial.")

    job_id = str(uuid.uuid4())
    job_dir = Path("/tmp/video-jobs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_name = safe_filename(video.filename, "input.mp4")
    input_video_path = job_dir / video_name

    output_name = f"{Path(video_name).stem}_cortado.mp4"
    output_path = job_dir / output_name

    await save_upload(video, input_video_path)

    command = ["ffmpeg", "-y"]

    if start_sec is not None:
        command += ["-ss", str(start_sec)]

    command += ["-i", str(input_video_path)]

    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            command += ["-t", str(duration)]
        else:
            command += ["-to", str(end_sec)]

    command += [
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    run_command(command, timeout=7200)

    background_tasks.add_task(shutil.rmtree, str(job_dir), ignore_errors=True)

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_name,
        background=background_tasks
    )

from fastapi import FastAPI
import subprocess

app = FastAPI()


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

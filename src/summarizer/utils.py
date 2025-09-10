import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from yt_dlp import YoutubeDL


def hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-\s_]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def sanitize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def yt_watch_base_url(video_id: Optional[str]) -> str:
    if not video_id:
        return "https://www.youtube.com/watch?"
    return f"https://www.youtube.com/watch?v={video_id}"


def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def yt_dlp_download_audio(url: str, outdir: Path) -> Tuple[Path, Dict]:
    """
    Download audio using yt-dlp and extract useful metadata.
    Returns (audio_path, metadata_dict).
    Requires ffmpeg installed on system.
    """
    outtmpl = str(outdir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "writethumbnail": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Audio file path
    vid = info.get("id")
    audio_path = outdir / f"{vid}.m4a"
    if not audio_path.exists():
        # fallback to original ext
        ext = info.get("ext", "m4a")
        alt = outdir / f"{vid}.{ext}"
        if alt.exists():
            audio_path = alt
        else:
            raise FileNotFoundError("Failed to locate downloaded audio file.")

    meta = {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url"),
    }
    return audio_path, meta
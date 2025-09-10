import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .utils import hhmmss


def transcribe_with_faster_whisper(
    audio_path: Path,
    model_size_or_path: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: Optional[str] = None
) -> Dict:
    """
    Transcribe using faster-whisper (local). Requires `pip install faster-whisper`.
    Returns:
      { language: str, segments: [ { start, end, start_timecode, end_timecode, text } ] }
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("faster-whisper is not installed. pip install faster-whisper") from e

    if device == "auto":
        # auto-select
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(str(audio_path), language=language, vad_filter=True)
    lang = info.language

    segments = []
    for seg in segments_iter:
        s = float(seg.start)
        e = float(seg.end)
        segments.append({
            "start": s,
            "end": e,
            "start_timecode": hhmmss(s),
            "end_timecode": hhmmss(e),
            "text": seg.text.strip()
        })

    return {
        "language": lang,
        "segments": segments
    }


def transcribe_with_openai_whisper(
    audio_path: Path,
    language: Optional[str] = None
) -> Dict:
    """
    Transcribe via OpenAI Whisper API. Requires OPENAI_API_KEY and openai>=1.0 (python SDK).
    Returns verbose segments if available.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed. pip install openai") from e

    client = OpenAI()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            language=language
        )

    # resp is a pydantic model; convert to dict and extract segments
    data = json.loads(resp.model_dump_json())
    lang = data.get("language") or "unknown"
    segments_api = data.get("segments") or []

    segments = []
    for seg in segments_api:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        txt = (seg.get("text") or "").strip()
        segments.append({
            "start": s,
            "end": e,
            "start_timecode": hhmmss(s),
            "end_timecode": hhmmss(e),
            "text": txt
        })

    # If segments missing (some responses only have text), fallback to a single segment
    if not segments and data.get("text"):
        segments.append({
            "start": 0.0,
            "end": 0.0,
            "start_timecode": "00:00:00",
            "end_timecode": "00:00:00",
            "text": (data["text"] or "").strip()
        })

    return {
        "language": lang,
        "segments": segments
    }
#!/usr/bin/env python3
import argparse
import dataclasses
import datetime
import json
import math
import os
import re
import sys
import tempfile
import textwrap
import uuid
from typing import List, Dict, Any, Optional, Tuple

# External deps:
# pip install yt-dlp langchain openai tiktoken pydantic pydantic-settings python-dotenv
# Choose one transcriber: pip install openai-whisper OR pip install faster-whisper
# ffmpeg must be installed on the system for yt-dlp/whisper

# LLM + LangChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Downloader
import yt_dlp

# Set defaults via env
DEFAULT_MODEL_SUMMARY = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
USE_OPENAI_TRANSCRIBE = (
    os.getenv("USE_OPENAI_TRANSCRIBE", "0") == "1"
)  # optional: OpenAI Whisper API
TRANSCRIBE_ENGINE = os.getenv(
    "TRANSCRIBE_ENGINE", "faster-whisper"
)  # "whisper" | "faster-whisper" | "openai"

# Try to import local whisper toolings
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

try:
    import whisper  # openai-whisper

    WHISPER_AVAILABLE = True
except Exception:
    pass

try:
    from faster_whisper import WhisperModel  # faster-whisper

    FASTER_WHISPER_AVAILABLE = True
except Exception:
    pass

# For OpenAI audio transcription API
from openai import OpenAI as OpenAIClient


@dataclasses.dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclasses.dataclass
class SummaryResult:
    url: str
    title: Optional[str]
    language: str
    executive_summary: str
    chapters: List[Dict[str, Any]]  # {title, start, end}
    key_takeaways: List[str]
    segments: List[TranscriptSegment]


def hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def download_youtube_audio(
    youtube_url: str, out_dir: str
) -> Tuple[str, Dict[str, Any]]:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    info = {}

    def _hook(d):
        pass

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.add_progress_hook(_hook)
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        filepath = os.path.join(out_dir, f"{video_id}.mp3")
        if not os.path.exists(filepath):
            # Sometimes yt-dlp uses audio extensions differently
            # Search for downloaded file
            for f in os.listdir(out_dir):
                if f.startswith(video_id) and f.endswith(".mp3"):
                    filepath = os.path.join(out_dir, f)
                    break
        if not os.path.exists(filepath):
            raise FileNotFoundError("Could not locate downloaded audio file.")
    return filepath, info


def transcribe_local_whisper(
    audio_path: str, model_size: str = "base"
) -> Tuple[List[TranscriptSegment], str]:
    if not WHISPER_AVAILABLE:
        raise RuntimeError("openai-whisper not installed. pip install openai-whisper")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=False, verbose=False)
    language = result.get("language", "unknown")
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            TranscriptSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=clean_text(seg["text"]),
            )
        )
    return segments, language


def transcribe_faster_whisper(
    audio_path: str, model_size: str = "base", device: Optional[str] = None
) -> Tuple[List[TranscriptSegment], str]:
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper not installed. pip install faster-whisper")
    if device is None:
        # auto-select device; faster-whisper uses "auto" implicitly when device not specified
        device = "auto"
    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments_iter, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    language = info.language if info and getattr(info, "language", None) else "unknown"
    segments: List[TranscriptSegment] = []
    for s in segments_iter:
        segments.append(
            TranscriptSegment(
                start=float(s.start),
                end=float(s.end),
                text=clean_text(s.text),
            )
        )
    return segments, language


def transcribe_openai_api(
    audio_path: str, model: str = "whisper-1"
) -> Tuple[List[TranscriptSegment], str]:
    client = OpenAIClient()
    # This uses OpenAI's transcription which typically returns text only. We’ll use word timestamps via SRT if available is limited.
    # To preserve timestamps, we can request verbose_json if supported; otherwise use chunking workaround is limited.
    # Below we try a JSON response with segments, falling back to plain text split if not available.
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=model, file=f, response_format="verbose_json"
        )
    language = getattr(transcription, "language", "unknown")
    segments: List[TranscriptSegment] = []
    segs = getattr(transcription, "segments", None)
    if segs:
        for seg in segs:
            segments.append(
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=clean_text(seg["text"]),
                )
            )
    else:
        # Fallback: single segment, no timestamps
        text = transcription.text if hasattr(transcription, "text") else ""
        segments.append(TranscriptSegment(start=0.0, end=0.0, text=clean_text(text)))
    return segments, language


def batch_transcript_segments(
    segments: List[TranscriptSegment], max_chars: int = 8000
) -> List[List[TranscriptSegment]]:
    # Batches transcript into chunks to fit LLM input limits while preserving segment boundaries.
    batches = []
    curr = []
    acc_len = 0
    for seg in segments:
        seg_len = len(seg.text) + 1
        if curr and acc_len + seg_len > max_chars:
            batches.append(curr)
            curr = [seg]
            acc_len = seg_len
        else:
            curr.append(seg)
            acc_len += seg_len
    if curr:
        batches.append(curr)
    return batches


def build_batch_text(batch: List[TranscriptSegment]) -> str:
    # Include timestamps to help LLM detect topical shifts
    lines = []
    for s in batch:
        lines.append(f"[{hhmmss(s.start)}–{hhmmss(s.end)}] {s.text}")
    return "\n".join(lines)


def summarize_with_chapters(
    url: str,
    title: Optional[str],
    language: str,
    segments: List[TranscriptSegment],
    model_name: str = DEFAULT_MODEL_SUMMARY,
    temperature: float = 0.3,
) -> SummaryResult:
    # Use batching for long transcripts
    batches = batch_transcript_segments(segments, max_chars=8000)
    batch_texts = [build_batch_text(b) for b in batches]

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # Step 1: Create consolidated abstractive summary and candidate chapters per batch
    per_batch_summaries = []
    per_batch_chapters = []
    per_batch_takeaways = []

    system_prompt = (
        "You are an expert at summarizing long transcripts and detecting topical shifts. "
        "You produce precise, timestamp-aligned chapters and concise summaries."
    )

    for i, bt in enumerate(batch_texts):
        user_prompt = f"""
You are given a transcript excerpt with timestamps. Tasks:
1) Write an abstractive summary (approx 120–180 words for this excerpt).
2) Propose 2–6 chapter segments for only this excerpt with:
   - title
   - start timestamp
   - end timestamp
   - brief one-line description
   Ensure timestamps are from the transcript lines and boundaries align with topical shifts.
3) List 3–5 bullet key takeaways for this excerpt.

Return JSON with keys: summary, chapters, takeaways.
Transcript excerpt:
{bt}
""".strip()

        resp = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        content = resp.content.strip()
        # Try to load JSON from content. If not JSON, attempt to extract JSON via simple heuristics.
        parsed = try_parse_json_from_text(content)
        if not parsed:
            parsed = {"summary": content[:800], "chapters": [], "takeaways": []}
        per_batch_summaries.append(parsed.get("summary", ""))
        per_batch_chapters.append(parsed.get("chapters", []))
        per_batch_takeaways.append(parsed.get("takeaways", []))

    # Step 2: Merge batch summaries and chapters into a cohesive final result
    merged_summary_text = "\n\n".join(per_batch_summaries)
    merged_takeaways = [item for sub in per_batch_takeaways for item in sub]
    # Normalize chapters and snap to nearest underlying transcript boundaries for accuracy
    normalized_chapters = normalize_and_snap_chapters(per_batch_chapters, segments)

    # Step 3: Ask LLM to produce final executive summary (200–300 words),
    # 5–12 chapters with final titles, and consolidated key takeaways.
    full_transcript_outline = build_outline_for_llm(segments, max_chars=12000)
    final_prompt = f"""
Context:
- Video URL: {url}
- Title: {title or "Unknown"}
- Language: {language}

You will produce a final deliverable with:
1) Executive summary (200–300 words).
2) 5–12 timestamped chapters (title + start + end), using detected topical shifts. Keep boundaries accurate.
3) 5–10 bullet-point key takeaways.

You are given:
A) Consolidated excerpt summaries:
{merged_summary_text}

B) Candidate chapters (already snapped to transcript boundaries):
{json.dumps(normalized_chapters, ensure_ascii=False, indent=2)}

C) Transcript outline with timestamps (for reference):
{full_transcript_outline}

Return JSON with keys: executive_summary, chapters, key_takeaways.
Chapters should be a list of objects with keys: title, start, end.
""".strip()

    resp2 = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=final_prompt)]
    )
    content2 = resp2.content.strip()
    final_parsed = try_parse_json_from_text(content2)
    if not final_parsed:
        final_parsed = {
            "executive_summary": content2[:1200],
            "chapters": [],
            "key_takeaways": merged_takeaways[:10],
        }

    # As a safety step, ensure chapters timestamps align with transcript segments
    aligned_chapters = align_chapters_to_segments(
        final_parsed.get("chapters", []), segments
    )

    return SummaryResult(
        url=url,
        title=title,
        language=language,
        executive_summary=final_parsed.get("executive_summary", ""),
        chapters=aligned_chapters,
        key_takeaways=final_parsed.get("key_takeaways", merged_takeaways[:10]),
        segments=segments,
    )


def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # Extract first JSON object in text using simple bracket matching
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def normalize_and_snap_chapters(
    per_batch_chapters: List[List[Dict[str, Any]]], segments: List[TranscriptSegment]
) -> List[Dict[str, Any]]:
    # Flatten, parse timestamps, snap to nearest transcript boundaries, and deduplicate overlaps
    flat = []
    for batch in per_batch_chapters:
        for ch in batch:
            title = clean_text(ch.get("title", ""))
            start = parse_ts_to_seconds(ch.get("start"))
            end = parse_ts_to_seconds(ch.get("end"))
            if start is None or end is None or end <= start:
                continue
            flat.append({"title": title, "start": start, "end": end})
    # Snap to segment boundaries
    snapped = align_chapters_to_segments(flat, segments)
    # Merge overlaps and reduce to reasonable number (will be refined by final pass)
    merged = merge_overlapping_chapters(snapped)
    return merged


def parse_ts_to_seconds(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    m = re.match(r"^(\d{1,2}):([0-5]?\d):([0-5]?\d)$", s)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2))
        se = int(m.group(3))
        return float(h * 3600 + mi * 60 + se)
    m2 = re.match(r"^([0-5]?\d):([0-5]?\d)$", s)
    if m2:
        mi = int(m2.group(1))
        se = int(m2.group(2))
        return float(mi * 60 + se)
    # fallback digits
    try:
        return float(s)
    except Exception:
        return None


def align_chapters_to_segments(
    chapters: List[Dict[str, Any]], segments: List[TranscriptSegment]
) -> List[Dict[str, Any]]:
    if not segments:
        return chapters
    seg_starts = [s.start for s in segments]
    seg_ends = [s.end for s in segments]

    def snap(t: float, to_start: bool) -> float:
        # snap timestamp to nearest existing segment boundary
        candidates = seg_starts if to_start else seg_ends
        closest = min(candidates, key=lambda x: abs(x - t))
        return closest

    aligned = []
    last_end = 0.0
    for ch in chapters:
        s = float(ch["start"])
        e = float(ch["end"])
        s_snapped = snap(s, True)
        e_snapped = snap(e, False)
        # ensure non-decreasing and positive duration
        s_snapped = max(s_snapped, last_end)
        if e_snapped <= s_snapped:
            # try to push to next segment end
            e_snapped = next((se for se in seg_ends if se > s_snapped), s_snapped + 1.0)
        aligned.append(
            {
                "title": clean_text(ch.get("title", "")),
                "start": s_snapped,
                "end": e_snapped,
            }
        )
        last_end = e_snapped
    # Ensure within video total duration (from segments)
    total_end = segments[-1].end
    for ch in aligned:
        ch["start"] = max(0.0, min(ch["start"], total_end))
        ch["end"] = max(ch["start"] + 0.5, min(ch["end"], total_end))
    # Filter zero/negative durations
    aligned = [c for c in aligned if c["end"] > c["start"]]
    return aligned


def merge_overlapping_chapters(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chapters:
        return []
    sorted_ch = sorted(chapters, key=lambda x: (x["start"], x["end"]))
    merged = [sorted_ch[0]]
    for ch in sorted_ch[1:]:
        prev = merged[-1]
        if ch["start"] <= prev["end"] - 1e-6:
            # overlap: merge by extending end and concatenating titles
            prev["end"] = max(prev["end"], ch["end"])
            if ch["title"] and ch["title"] not in prev["title"]:
                prev["title"] = clean_text(prev["title"] + " / " + ch["title"])
        else:
            merged.append(ch)
    return merged


def build_outline_for_llm(
    segments: List[TranscriptSegment], max_chars: int = 12000
) -> str:
    # Provide skeleton of transcript with timestamps every N segments
    lines = []
    acc = 0
    for i, s in enumerate(segments):
        line = f"[{hhmmss(s.start)}–{hhmmss(s.end)}] {s.text}"
        if acc + len(line) > max_chars:
            break
        lines.append(line)
        acc += len(line)
    return "\n".join(lines)


def segments_to_text(segments: List[TranscriptSegment]) -> str:
    return "\n".join(f"[{hhmmss(s.start)}–{hhmmss(s.end)}] {s.text}" for s in segments)


def to_markdown(result: SummaryResult) -> str:
    md = []
    md.append(f"# Summary for: {result.title or 'Untitled'}")
    md.append(f"- URL: {result.url}")
    md.append(f"- Language: {result.language}")
    md.append("")
    md.append("## Executive Summary")
    md.append(result.executive_summary.strip())
    md.append("")
    md.append("## Chapters")
    for ch in result.chapters:
        md.append(f"- [{hhmmss(ch['start'])} – {hhmmss(ch['end'])}] {ch['title']}")
    md.append("")
    md.append("## Key Takeaways")
    for k in result.key_takeaways:
        md.append(f"- {k}")
    md.append("")
    md.append("## Transcript (timestamped)")
    md.append(segments_to_text(result.segments))
    return "\n".join(md)


def transcribe(audio_path: str) -> Tuple[List[TranscriptSegment], str]:
    engine = TRANSCRIBE_ENGINE.lower()
    if USE_OPENAI_TRANSCRIBE or engine == "openai":
        return transcribe_openai_api(audio_path)
    elif engine == "faster-whisper":
        return transcribe_faster_whisper(
            audio_path, model_size=os.getenv("FASTER_WHISPER_MODEL", "base")
        )
    elif engine == "whisper":
        return transcribe_local_whisper(
            audio_path, model_size=os.getenv("WHISPER_MODEL", "base")
        )
    else:
        # default fallback
        if FASTER_WHISPER_AVAILABLE:
            return transcribe_faster_whisper(audio_path, model_size="base")
        elif WHISPER_AVAILABLE:
            return transcribe_local_whisper(audio_path, model_size="base")
        else:
            raise RuntimeError(
                "No transcription engine available. Install faster-whisper or openai-whisper, or set USE_OPENAI_TRANSCRIBE=1."
            )


def run_pipeline(url: str, model_name: str = DEFAULT_MODEL_SUMMARY) -> SummaryResult:
    with tempfile.TemporaryDirectory() as td:
        audio_path, info = download_youtube_audio(url, td)
        title = info.get("title")
        segments, language = transcribe(audio_path)
        result = summarize_with_chapters(
            url=url,
            title=title,
            language=language,
            segments=segments,
            model_name=model_name,
        )
        return result


def save_outputs(result: SummaryResult, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", (result.title or "summary"))[:60]
    json_path = os.path.join(out_dir, f"{base}.json")
    md_path = os.path.join(out_dir, f"{base}.md")

    obj = {
        "url": result.url,
        "title": result.title,
        "language": result.language,
        "executive_summary": result.executive_summary,
        "chapters": [
            {
                "title": ch["title"],
                "start": ch["start"],
                "end": ch["end"],
                "start_hhmmss": hhmmss(ch["start"]),
                "end_hhmmss": hhmmss(ch["end"]),
            }
            for ch in result.chapters
        ],
        "key_takeaways": result.key_takeaways,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "start_hhmmss": hhmmss(s.start),
                "end_hhmmss": hhmmss(s.end),
            }
            for s in result.segments
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(result))
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(
        description="YouTube/video summarizer with chapters"
    )
    parser.add_argument("youtube_url", help="YouTube URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_SUMMARY,
        help="OpenAI model for summarization (e.g., gpt-4o-mini)",
    )
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()

    res = run_pipeline(args.youtube_url, model_name=args.model)
    json_path, md_path = save_outputs(res, args.out)
    print(f"Executive summary:\n{textwrap.fill(res.executive_summary, width=100)}\n")
    print("Chapters:")
    for ch in res.chapters:
        print(f"- {hhmmss(ch['start'])} – {hhmmss(ch['end'])}: {ch['title']}")
    print("\nKey takeaways:")
    for k in res.key_takeaways:
        print(f"- {k}")
    print(f"\nSaved:\n- {json_path}\n- {md_path}")


if __name__ == "__main__":
    main()

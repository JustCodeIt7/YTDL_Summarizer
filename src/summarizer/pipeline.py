import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .transcribe import transcribe_with_faster_whisper, transcribe_with_openai_whisper
from .utils import (
    hhmmss,
    sanitize_text,
    slugify,
    yt_dlp_download_audio,
    yt_watch_base_url,
)


@dataclass
class PipelineConfig:
    transcriber: str = "faster-whisper"  # or "openai"
    whisper_model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    llm_model: str = "gpt-4o-mini"
    min_chapters: int = 6
    max_chapters: int = 10
    force_language: Optional[str] = None
    # summarization chunking
    chunk_size_chars: int = 6000
    chunk_overlap_chars: int = 400


class SummarizationPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.llm = ChatOpenAI(model=cfg.llm_model, temperature=0.2)

    def process(self, youtube_url: str, progress_cb: Optional[Callable] = None) -> Dict:
        progress = lambda x: None
        if progress_cb:
            def progress(x):
                try:
                    progress_cb(x)
                except Exception:
                    pass

        progress(0)
        # 1) Download audio + metadata
        tmp_dir = Path(tempfile.mkdtemp(prefix="yt-summarizer-"))
        audio_path, meta = yt_dlp_download_audio(youtube_url, tmp_dir)
        progress(10)

        # 2) Transcribe
        if self.cfg.transcriber == "faster-whisper":
            transcript = transcribe_with_faster_whisper(
                audio_path,
                model_size_or_path=self.cfg.whisper_model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                language=self.cfg.force_language
            )
        elif self.cfg.transcriber == "openai":
            transcript = transcribe_with_openai_whisper(
                audio_path,
                language=self.cfg.force_language
            )
        else:
            raise ValueError(f"Unknown transcriber: {self.cfg.transcriber}")
        progress(40)

        # 3) Build chapters with topical shift detection
        chapters = self._build_chapters(transcript["segments"], meta["duration"], youtube_url, desired_range=(self.cfg.min_chapters, self.cfg.max_chapters))
        progress(60)

        # 4) Summaries (executive + takeaways)
        full_text = " ".join(seg["text"] for seg in transcript["segments"])
        executive_summary = self._summarize_text(full_text, language=transcript["language"])
        key_takeaways = self._key_takeaways(full_text, language=transcript["language"])
        progress(85)

        # 5) Title each chapter via LLM
        chapters = self._title_chapters(chapters, transcript["segments"], language=transcript["language"])
        progress(95)

        result = {
            "metadata": {
                "video_url": youtube_url,
                "title": meta.get("title"),
                "uploader": meta.get("uploader"),
                "video_id": meta.get("id"),
                "duration": meta.get("duration"),
                "watch_base_url": yt_watch_base_url(meta.get("id")),
            },
            "transcript": transcript,
            "summary": {
                "executive_summary": executive_summary,
                "key_takeaways": key_takeaways,
            },
            "chapters": chapters,
            "artifacts": {
                "audio_path": str(audio_path),
            }
        }

        # Cleanup temp dir but leave audio by default (caller may move/delete)
        progress(100)
        try:
            # keep the downloaded audio path so caller can move; tmp dir may still be deletable if audio moved
            pass
        except Exception:
            pass

        return result

    def _build_chapters(
        self,
        segments: List[Dict],
        total_duration: Optional[float],
        video_url: str,
        desired_range: Tuple[int, int] = (6, 10),
    ) -> List[Dict]:
        """
        Detect topical shifts and build 5–12 chapters with timestamps aligned to transcript segments.
        Algorithm:
          - Merge segments into ~30s windows to reduce noise
          - Use TF-IDF cosine similarity between adjacent windows
          - Find boundaries at local minima (low similarity) with dynamic threshold
          - Select top-k boundaries to fit desired chapter count range
        """
        # Merge to ~30s windows
        target_win = 30.0
        windows = []
        cur = None
        for seg in segments:
            s, e, txt = float(seg["start"]), float(seg["end"]), seg["text"]
            if cur is None:
                cur = {"start": s, "end": e, "text": txt}
            else:
                if (cur["end"] - cur["start"]) < target_win:
                    cur["end"] = e
                    cur["text"] += " " + txt
                else:
                    windows.append(cur)
                    cur = {"start": s, "end": e, "text": txt}
        if cur:
            windows.append(cur)
        if len(windows) < 2:
            # fallback: single chapter
            start = float(segments[0]["start"]) if segments else 0.0
            end = float(segments[-1]["end"]) if segments else (total_duration or 0.0)
            return [{
                "idx": 0,
                "title": "Full Video",
                "start": start,
                "end": end,
                "start_timecode": hhmmss(start),
                "end_timecode": hhmmss(end),
                "url": f"{yt_watch_base_url_from_url(video_url)}&t={int(start)}s",
            }]

        # TF-IDF
        texts = [sanitize_text(w["text"]) for w in windows]
        vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        X = vectorizer.fit_transform(texts)
        # adjacency similarity
        sims = []
        for i in range(len(windows) - 1):
            sim = cosine_similarity(X[i], X[i + 1])[0, 0]
            sims.append(sim)

        # Score boundaries where similarity dips (low sim => high boundary score)
        # Normalize to 0..1
        if sims:
            import numpy as np
            s = np.array(sims)
            # invert similarity
            inv = 1.0 - s
            # smooth
            kernel = np.array([0.25, 0.5, 0.25])
            inv_smooth = np.convolve(inv, kernel, mode="same")
            # dynamic threshold: top-N peaks to get desired chapters
            desired_min, desired_max = desired_range
            desired = min(max(desired_min, 5), desired_max)
            desired = min(desired, len(windows))  # cannot exceed windows
            # number of chapters = boundaries + 1; choose k = desired - 1 boundaries
            k = max(0, desired - 1)
            # pick top-k peaks with minimal separation of 2 windows
            peak_idx = inv_smooth.argsort()[::-1].tolist()

            chosen = []
            for idx in peak_idx:
                if len(chosen) >= k:
                    break
                # enforce minimal separation and min chapter duration
                if all(abs(idx - c) >= 2 for c in chosen):
                    chosen.append(idx)
            chosen = sorted(chosen)

            # Build chapters from boundaries
            boundaries = [0] + [i + 1 for i in chosen] + [len(windows)]
            chapters = []
            for ci in range(len(boundaries) - 1):
                wi = boundaries[ci]
                wj = boundaries[ci + 1] - 1
                start = windows[wi]["start"]
                end = windows[wj]["end"]
                chapters.append({
                    "idx": ci,
                    "title": f"Chapter {ci+1}",
                    "start": float(start),
                    "end": float(end),
                    "start_timecode": hhmmss(start),
                    "end_timecode": hhmmss(end),
                    "url": None,  # filled after titling
                })
        else:
            # only one window pair -> two chapters
            chapters = []
            start1 = windows[0]["start"]
            end1 = windows[0]["end"]
            start2 = windows[1]["start"]
            end2 = windows[-1]["end"]
            chapters.append({
                "idx": 0,
                "title": "Chapter 1",
                "start": float(start1),
                "end": float(end1),
                "start_timecode": hhmmss(start1),
                "end_timecode": hhmmss(end1),
                "url": None,
            })
            chapters.append({
                "idx": 1,
                "title": "Chapter 2",
                "start": float(start2),
                "end": float(end2),
                "start_timecode": hhmmss(start2),
                "end_timecode": hhmmss(end2),
                "url": None,
            })

        # Clip and ensure monotonic increasing timestamps
        base_url = yt_watch_base_url_from_url(video_url)
        for ch in chapters:
            ch["url"] = f"{base_url}&t={int(ch['start'])}s"

        return chapters

    def _summarize_text(self, text: str, language: str) -> str:
        """
        Map-reduce style summarization: chunk transcript, summarize chunks, then synthesize 200–300 words.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size_chars,
            chunk_overlap=self.cfg.chunk_overlap_chars
        )
        chunks = splitter.split_text(text)
        # Map
        map_summaries = []
        for chunk in chunks:
            prompt = (
                f"You are a helpful assistant that summarizes a transcript in {language or 'the original language'}.\n\n"
                "Summarize the following part of a longer transcript succinctly. Focus on the main points, definitions, and steps. "
                "Return 4–7 bullet points, concise and faithful to the content.\n\n"
                f"Transcript chunk:\n'''{chunk}'''\n\n"
                "Bulleted summary:"
            )
            resp = self.llm.invoke(prompt)
            map_summaries.append(resp.content.strip())
        # Reduce
        reduce_prompt = (
            "You are an expert editor. Given bullet summaries of different parts of a video transcript, "
            "compose a cohesive executive summary of the entire video in 200–300 words. "
            "Write in the same language as the original transcript (if known). Avoid repetition, keep it high-level and actionable.\n\n"
            f"Language hint: {language or 'unknown'}\n\n"
            "Bulleted summaries of chunks:\n"
            + "\n\n".join(map_summaries)
            + "\n\nExecutive summary (200–300 words):"
        )
        reduce_resp = self.llm.invoke(reduce_prompt)
        return reduce_resp.content.strip()

    def _key_takeaways(self, text: str, language: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size_chars,
            chunk_overlap=self.cfg.chunk_overlap_chars
        )
        chunks = splitter.split_text(text)
        # Get high-signal bullets from each chunk then distill
        bullets_all = []
        for chunk in chunks:
            prompt = (
                f"Extract the 3–6 most important, concrete, and non-redundant key takeaways from this transcript chunk. "
                f"Write them as short bullet points in {language or 'the original language'}.\n\n"
                f"Transcript chunk:\n'''{chunk}'''\n\n"
                "Key takeaways:"
            )
            resp = self.llm.invoke(prompt)
            bullets_all.append(resp.content.strip())

        reduce_prompt = (
            "You are an expert editor. Consolidate the key takeaways from all chunks into 6–10 distinct bullet points. "
            "Remove redundancy, keep each bullet concise and actionable. Preserve the original language.\n\n"
            "Bullets from chunks:\n"
            + "\n\n".join(bullets_all)
            + "\n\nFinal consolidated key takeaways (6–10 bullets):"
        )
        resp = self.llm.invoke(reduce_prompt)
        # Normalize to a list
        lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in resp.content.splitlines()]
        lines = [l for l in lines if l]
        return lines[:10]

    def _title_chapters(
        self,
        chapters: List[Dict],
        segments: List[Dict],
        language: str
    ) -> List[Dict]:
        # Build text per chapter by concatenating contained segments
        chapter_texts = []
        for ch in chapters:
            text = []
            for seg in segments:
                s, e = float(seg["start"]), float(seg["end"])
                if s >= ch["start"] and e <= ch["end"]:
                    text.append(seg["text"])
            chapter_texts.append(" ".join(text).strip())

        titled = []
        for ch, text in zip(chapters, chapter_texts):
            # Keep prompt small by clipping text
            sample = text[:4000]
            prompt = (
                f"Create a concise, informative chapter title (max 60 characters) in {language or 'the original language'}. "
                "Do not include timestamps or numbering. Return only the title text.\n\n"
                f"Chapter transcript excerpt:\n'''{sample}'''\n\n"
                "Title:"
            )
            try:
                resp = self.llm.invoke(prompt)
                title = resp.content.strip().strip('"').strip("'")
                if not title:
                    title = f"Chapter {ch['idx']+1}"
            except Exception:
                title = f"Chapter {ch['idx']+1}"
            ch2 = dict(ch)
            ch2["title"] = title
            titled.append(ch2)
        return titled

    def to_markdown(self, result: Dict) -> str:
        """
        Render a human-friendly Markdown report including deep-links to timestamps.
        """
        meta = result["metadata"]
        chapters = result["chapters"]
        summary = result["summary"]
        transcript = result["transcript"]

        video_url = meta["video_url"]
        base_url = meta["watch_base_url"]

        lines = []
        lines.append(f"# {meta.get('title') or 'Video Summary'}")
        lines.append("")
        lines.append(f"- URL: {video_url}")
        if meta.get("uploader"):
            lines.append(f"- Uploader: {meta['uploader']}")
        if meta.get("duration"):
            lines.append(f"- Duration: {hhmmss(meta['duration'])}")
        if transcript.get("language"):
            lines.append(f"- Language: {transcript['language']}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(summary["executive_summary"])
        lines.append("")
        lines.append("## Chapters")
        lines.append("")
        for ch in chapters:
            lines.append(f"- [{ch['start_timecode']}] [{ch['title']}]({base_url}&t={int(ch['start'])}s)")
        lines.append("")
        lines.append("## Key Takeaways")
        lines.append("")
        for k in summary["key_takeaways"]:
            lines.append(f"- {k}")
        lines.append("")
        lines.append("## Transcript (segments)")
        lines.append("")
        for seg in transcript["segments"]:
            lines.append(f"- [{seg['start_timecode']} - {seg['end_timecode']}] {seg['text']}")
        lines.append("")
        return "\n".join(lines)


def yt_watch_base_url_from_url(url: str) -> str:
    """
    Derive watch base URL with video id only, to append &t= later.
    """
    # Use utils version if possible
    vid = None
    m = re.search(r"v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        vid = m.group(1)
    if not vid:
        # Short URLs
        m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
        if m:
            vid = m.group(1)
    return yt_watch_base_url(vid or "")
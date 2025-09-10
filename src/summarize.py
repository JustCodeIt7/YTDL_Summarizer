#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import time

from summarizer.pipeline import SummarizationPipeline, PipelineConfig
from summarizer.utils import slugify, write_text_file


def main():
    parser = argparse.ArgumentParser(
        description="YouTube/video summarizer with chapters (CLI)"
    )
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument(
        "--transcriber",
        choices=["faster-whisper", "openai"],
        default="faster-whisper",
        help="Choose transcription backend"
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper/local whisper model size/name (e.g., tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for faster-whisper: auto, cpu, cuda"
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type for faster-whisper: auto, int8, int8_float16, float16, float32"
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        help="OpenAI Chat model for summarization (e.g., gpt-4o-mini, gpt-4o, gpt-4.1-mini)"
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=10,
        help="Maximum number of chapters to generate (5–12 recommended)"
    )
    parser.add_argument(
        "--min-chapters",
        type=int,
        default=6,
        help="Minimum number of chapters to generate (5–12 recommended)"
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory to write outputs"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep downloaded audio file"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (auto-detect if not provided)"
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        transcriber=args.transcriber,
        whisper_model=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        llm_model=args.llm_model,
        min_chapters=args.min_chapters,
        max_chapters=args.max_chapters,
        force_language=args.language
    )

    t0 = time.time()
    pipeline = SummarizationPipeline(cfg)

    try:
        result = pipeline.process(args.youtube_url)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Prepare output directory
    video_id = result["metadata"]["video_id"]
    title = result["metadata"]["title"] or video_id
    slug = slugify(title) or video_id
    outdir = Path(args.outdir) / slug
    outdir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = outdir / "summary.json"
    json.dump(result, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Write Markdown
    md_text = pipeline.to_markdown(result)
    md_path = outdir / "summary.md"
    write_text_file(md_path, md_text)

    # Optionally move audio file into output dir
    if result.get("artifacts", {}).get("audio_path"):
        audio_path = Path(result["artifacts"]["audio_path"])
        if audio_path.exists():
            target = outdir / audio_path.name
            if str(audio_path) != str(target):
                try:
                    audio_path.replace(target)
                    result["artifacts"]["audio_path"] = str(target)
                except Exception:
                    pass
        if not args.keep_audio:
            try:
                Path(result["artifacts"]["audio_path"]).unlink(missing_ok=True)
            except Exception:
                pass

    elapsed = time.time() - t0

    # Print a concise CLI output
    print(f"\nTitle: {title}")
    print(f"URL: {result['metadata']['video_url']}")
    print(f"Language: {result['transcript']['language']}")
    print("\nExecutive summary:")
    print(result["summary"]["executive_summary"])
    print("\nChapters:")
    for ch in result["chapters"]:
        print(f"- [{ch['start_timecode']}] {ch['title']}")

    print(f"\nKey takeaways:")
    for item in result["summary"]["key_takeaways"]:
        print(f"- {item}")

    print(f"\nWrote:")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
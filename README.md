# YouTube Video Summarizer with Chapters (CLI + Web)

A Python tool to:

- Download a YouTube video's audio
- Transcribe it with Whisper (local via faster-whisper or via OpenAI Whisper API)
- Detect topical shifts to create timestamped chapters
- Generate an executive summary (200–300 words) and key takeaways using OpenAI via LangChain
- Export JSON and Markdown
- Use as a CLI or a Streamlit web app

## Stack

- Python
- yt-dlp for audio download (requires `ffmpeg` installed)
- Transcription:
  - Local: faster-whisper (preferred for speed and accuracy)
  - API: OpenAI Whisper API
- LangChain + OpenAI for LLM summarization and titling
- Streamlit for simple web UI

## Features

- Input: YouTube URL
- Download audio, transcribe into segments with timestamps
- Generate:
  - Executive summary (200–300 words)
  - 5–12 chapter ranges with titles and timestamps (topic-shift detection on transcript windows)
  - Bullet-point key takeaways
- Export JSON and Markdown; display in web UI
- Constraints handled:
  - Long videos batched for summarization (map-reduce over transcript chunks)
  - Model choice (local `faster-whisper` vs. OpenAI Whisper API)
  - Automatic language detection (from Whisper)
- Acceptance criteria:
  - CLI: `python summarize.py <youtube_url>`
  - Web: `/summarize?url=...` shows summary and chapters (Streamlit accepts path segments; query param triggers auto-run)
  - Accurate timestamps aligned to transcript segments/windows

## Installation

1. System requirement: ffmpeg

- macOS (brew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
- Windows: Install from https://ffmpeg.org/download.html and add to PATH

2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. OpenAI credentials

- Set your API key in the environment:
  - macOS/Linux: `export OPENAI_API_KEY=your_key`
  - Windows (PowerShell): `$Env:OPENAI_API_KEY = "your_key"`

Optionally set your preferred chat model:

```bash
export OPENAI_LLM_MODEL=gpt-4o-mini
```

## Usage

### CLI

```bash
python summarize.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional flags:

- `--transcriber {faster-whisper|openai}` (default: faster-whisper)
- `--whisper-model small` (for faster-whisper: tiny/base/small/medium/large-v3 or a local path)
- `--device {auto|cpu|cuda}` and `--compute-type {auto|int8|int8_float16|float16|float32}`
- `--llm-model gpt-4o-mini` (OpenAI chat model)
- `--min-chapters 6 --max-chapters 10`
- `--language <lang_code>` to force language (e.g., `en`, `es`) if needed
- `--outdir outputs`
- `--keep-audio` to keep the downloaded audio file

Outputs are saved under `outputs/<video-slug>/summary.json` and `summary.md`.

### Web (Streamlit)

```bash
streamlit run streamlit_app.py
```

Open:

- http://localhost:8501/
- For auto-run via URL: http://localhost:8501/summarize?url=https://www.youtube.com/watch?v=VIDEO_ID

Notes:

- Streamlit serves one app at a single route; `/summarize` is accepted as a path segment and the app reads the `url` query param to auto-run.

## How it works

1. Download audio with `yt-dlp` (m4a) and extract metadata such as title, duration, and video ID.

2. Transcribe:

- Local (default): `faster-whisper` with VAD filtering; returns segments with timestamps and detected language.
- OpenAI Whisper API: Uses verbose JSON to retrieve segments with timestamps.

3. Chapter detection:

- Transcript segments are merged into ~30s windows to smooth noise.
- TF-IDF vectors measure cosine similarity between adjacent windows.
- Low-similarity points become boundaries indicating topical shifts.
- The algorithm picks top-k boundaries to produce 5–12 chapters, ensuring spacing and segment alignment.
- Chapter start/end timestamps align with nearest transcript windows.

4. Summarization:

- Map-reduce over transcript chunks with LangChain + OpenAI:
  - Map: chunk summaries (bulleted)
  - Reduce: cohesive 200–300 word executive summary
- Key takeaways: extract from chunks then consolidate to 6–10 bullets.
- Chapter titling: LLM generates a concise title for each chapter based on its text.

5. Export:

- JSON: metadata, transcript segments, chapters, summary, and key takeaways.
- Markdown: nicely formatted report with deep links to YouTube timestamps.

## JSON Schema (high-level)

```json
{
  "metadata": {
    "video_url": "...",
    "title": "...",
    "uploader": "...",
    "video_id": "...",
    "duration": 1234,
    "watch_base_url": "https://www.youtube.com/watch?v=ID"
  },
  "transcript": {
    "language": "en",
    "segments": [
      {
        "start": 0.0,
        "end": 4.2,
        "start_timecode": "00:00:00",
        "end_timecode": "00:00:04",
        "text": "..."
      }
    ]
  },
  "summary": {
    "executive_summary": "...",
    "key_takeaways": ["...", "..."]
  },
  "chapters": [
    {
      "idx": 0,
      "title": "Intro",
      "start": 0.0,
      "end": 120.5,
      "start_timecode": "00:00:00",
      "end_timecode": "00:02:00",
      "url": "https://www.youtube.com/watch?v=ID&t=0s"
    }
  ],
  "artifacts": {
    "audio_path": "/path/to/file.m4a"
  }
}
```

## Notes and Tips

- Large videos: The tool chunks transcripts to keep LLM context under limits; it then synthesizes a final summary and key takeaways.
- Faster-whisper models: choose `tiny`/`base` for speed, `small`/`medium` for balance, `large-v3` for best accuracy (slower).
- GPU: If you have CUDA, set `--device cuda` for faster-whisper.
- Timestamps: Chapter start times link directly to `&t=SECONDS` on the YouTube watch URL.
- If you see ffmpeg errors, ensure ffmpeg is installed and in PATH.

## License

MIT

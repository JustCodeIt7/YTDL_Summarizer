import io
import json
import os
from pathlib import Path
import streamlit as st

from summarizer.pipeline import SummarizationPipeline, PipelineConfig
from summarizer.utils import write_text_file

st.set_page_config(page_title="YouTube Video Summarizer with Chapters", layout="wide")

st.title("YouTube Video Summarizer with Chapters")

# Query param handling to support /summarize?url=...
query_params = st.query_params
prefill_url = query_params.get("url", "")

with st.sidebar:
    st.header("Settings")
    transcriber = st.selectbox(
        "Transcriber",
        options=["faster-whisper", "openai"],
        index=0
    )
    whisper_model = st.text_input(
        "Whisper model (for faster-whisper)",
        value="small",
        help="tiny/base/small/medium/large-v3 or a local path"
    )
    device = st.selectbox(
        "Device (faster-whisper)",
        options=["auto", "cpu", "cuda"],
        index=0
    )
    compute_type = st.selectbox(
        "Compute type (faster-whisper)",
        options=["auto", "int8", "int8_float16", "float16", "float32"],
        index=0
    )
    llm_model = st.text_input(
        "OpenAI Chat model",
        value=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
    )
    min_chapters = st.slider("Min chapters", 5, 12, 6)
    max_chapters = st.slider("Max chapters", 5, 12, 10)
    force_language = st.text_input("Force language code (optional)", value="")

    st.caption("Set OPENAI_API_KEY in your environment to use OpenAI models or Whisper API.")

url = st.text_input("YouTube URL", value=prefill_url, placeholder="https://www.youtube.com/watch?v=...")

run = st.button("Summarize") or (prefill_url and not st.session_state.get("auto_ran"))

if prefill_url and not st.session_state.get("auto_ran"):
    st.session_state["auto_ran"] = True

if run and url:
    cfg = PipelineConfig(
        transcriber=transcriber,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        llm_model=llm_model,
        min_chapters=min_chapters,
        max_chapters=max_chapters,
        force_language=force_language or None
    )
    pipe = SummarizationPipeline(cfg)

    with st.spinner("Downloading, transcribing, and summarizing..."):
        try:
            result = pipe.process(url, progress_cb=st.progress)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader(result["metadata"]["title"] or "Untitled")
    st.write(f"URL: {result['metadata']['video_url']}")
    st.write(f"Language: {result['transcript']['language']}")

    st.subheader("Executive Summary")
    st.write(result["summary"]["executive_summary"])

    st.subheader("Chapters")
    for ch in result["chapters"]:
        st.markdown(f"- [{ch['start_timecode']}] [{ch['title']}]({result['metadata']['watch_base_url']}&t={int(ch['start'])}s)")

    st.subheader("Key Takeaways")
    st.markdown("\n".join([f"- {k}" for k in result["summary"]["key_takeaways"]]))

    # Downloads
    st.subheader("Downloads")
    json_bytes = io.BytesIO(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("Download JSON", json_bytes, file_name="summary.json", mime="application/json")

    md_text = pipe.to_markdown(result)
    st.download_button("Download Markdown", md_text, file_name="summary.md", mime="text/markdown")

    # Raw transcript preview
    with st.expander("Transcript segments (preview)"):
        st.write(f"{len(result['transcript']['segments'])} segments")
        for seg in result["transcript"]["segments"][:200]:
            st.write(f"[{seg['start_timecode']} - {seg['end_timecode']}] {seg['text']}")
        if len(result["transcript"]["segments"]) > 200:
            st.caption("Showing first 200 segments.")


st.caption("Tip: You can open this app as /summarize?url=YOUR_URL to auto-run.")
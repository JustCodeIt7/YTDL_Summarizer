import os
import tempfile
import logging
from datetime import timedelta
import json
import streamlit as st
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv

# --- Configuration ---
# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# --- Core Logic ---


class VideoSummarizer:
    """
    A class to summarize YouTube videos by transcribing audio and using an LLM.
    """

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initializes the summarizer with Whisper and OpenAI models.
        """
        logging.info("Initializing models...")
        # For better performance on compatible hardware, you can change device to "cuda"
        # and compute_type to "float16".
        self.whisper_model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        logging.info("Models initialized successfully.")

    def _download_audio(self, url: str, temp_dir: str) -> str:
        """
        Downloads audio from a YouTube URL to a temporary directory.
        Returns the path to the downloaded audio file.
        """
        logging.info(f"Downloading audio from URL: {url}")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])
            if error_code != 0:
                raise Exception(f"Failed to download audio. Error code: {error_code}")
            info = ydl.extract_info(url, download=False)
            return os.path.join(temp_dir, f"{info['id']}.mp3")

    def _transcribe_audio(self, audio_path: str) -> tuple:
        """
        Transcribes the audio file using the Whisper model.
        Returns the transcription segments and language information.
        """
        logging.info(f"Transcribing audio file: {audio_path}")
        segments, info = self.whisper_model.transcribe(
            audio_path, word_timestamps=False
        )
        logging.info(
            f"Detected language '{info.language}' with probability {info.language_probability}"
        )
        return list(segments), info

    def _format_timestamp(self, seconds: float) -> str:
        """
        Formats seconds into HH:MM:SS format.
        """
        return str(timedelta(seconds=int(seconds)))

    def _group_segments_for_chapters(
        self, segments: list, min_chapters: int = 5, max_chapters: int = 12
    ) -> list:
        """
        Groups transcript segments into a target number of chapters.
        """
        total_duration = segments[-1].end if segments else 0
        num_chapters = min(
            max(int(total_duration / 180), min_chapters), max_chapters
        )  # Aim for a chapter every ~3 minutes

        if num_chapters <= 1:  # Handle very short videos
            return (
                [
                    {
                        "start": segments[0].start,
                        "end": segments[-1].end,
                        "segments": segments,
                    }
                ]
                if segments
                else []
            )

        chapter_duration = total_duration / num_chapters

        chapters = []
        current_chapter_segments = []
        current_chapter_start = segments[0].start if segments else 0

        for seg in segments:
            current_chapter_segments.append(seg)
            if (
                seg.end - current_chapter_start > chapter_duration
                and len(chapters) < num_chapters - 1
            ):
                chapters.append(
                    {
                        "start": current_chapter_start,
                        "end": seg.end,
                        "segments": current_chapter_segments,
                    }
                )
                current_chapter_segments = []
                if segments.index(seg) + 1 < len(segments):
                    current_chapter_start = segments[segments.index(seg) + 1].start

        if current_chapter_segments:
            chapters.append(
                {
                    "start": current_chapter_start,
                    "end": current_chapter_segments[-1].end,
                    "segments": current_chapter_segments,
                }
            )

        return chapters

    def generate_summary(self, youtube_url: str) -> dict:
        """
        The main method to generate a full summary from a YouTube URL.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._download_audio(youtube_url, temp_dir)
            segments, info = self._transcribe_audio(audio_path)

            docs = [
                Document(page_content=seg.text, metadata={"start": seg.start})
                for seg in segments
            ]

            logging.info("Generating executive summary...")
            summary_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            executive_summary = summary_chain.run(docs)

            logging.info("Generating chapters...")
            chapter_chunks = self._group_segments_for_chapters(segments)
            chapters = []
            title_prompt = PromptTemplate.from_template(
                "You are an expert at creating concise, descriptive titles for video segments. "
                "Based on the following transcript segment, what is the best title? "
                "Provide only the title itself, without any prefixes like 'Title:'.\n\n"
                'Transcript: "{text}"\n\nTITLE:'
            )
            for chunk in chapter_chunks:
                chunk_text = " ".join(seg.text for seg in chunk["segments"])
                title = self.llm.invoke(
                    title_prompt.format(text=chunk_text)
                ).content.strip()
                timestamp = self._format_timestamp(chunk["start"])
                chapters.append({"timestamp": timestamp, "title": title})

            logging.info("Generating key takeaways...")
            takeaways_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="refine",
                question_prompt=PromptTemplate.from_template(
                    'Analyze the following text from a video transcript and extract the most important takeaways as a bulleted list.\n\nTEXT: "{text}"\n\nKEY TAKEAWAYS:'
                ),
                refine_prompt=PromptTemplate.from_template(
                    "Here is a list of existing takeaways: {existing_answer}\n"
                    'We have more context from the transcript below. Refine the existing list by adding any new key takeaways from the new context.\n\nCONTEXT: "{text}"\n\nREFINED TAKEAWAYS:'
                ),
            )
            takeaways_result = takeaways_chain.run(docs)
            key_takeaways = [
                item.strip().lstrip("- ")
                for item in takeaways_result.strip().split("\n")
                if item.strip()
            ]

            return {
                "executive_summary": executive_summary,
                "chapters": chapters,
                "key_takeaways": key_takeaways,
                "metadata": {"url": youtube_url, "language": info.language},
            }


def to_markdown(summary_data: dict, url: str) -> str:
    """Formats the summary data into a Markdown string."""
    lines = [f"# Summary for [YouTube Video]({url})\n"]
    lines.append("## 📝 Executive Summary")
    lines.append(summary_data["executive_summary"])
    lines.append("\n---\n")
    lines.append("## 📖 Chapters")
    for chapter in summary_data["chapters"]:
        h, m, s = map(int, chapter["timestamp"].split(":"))
        total_seconds = h * 3600 + m * 60 + s
        chapter_url = f"{url}&t={total_seconds}s"
        lines.append(
            f"- [{chapter['timestamp']}]({chapter_url}) - **{chapter['title']}**"
        )
    lines.append("\n---\n")
    lines.append("## 🔑 Key Takeaways")
    for takeaway in summary_data["key_takeaways"]:
        lines.append(f"- {takeaway}")
    return "\n".join(lines)


# --- Streamlit Web App ---


# Caching function for the summarizer model
@st.cache_resource
def get_summarizer():
    """Creates and returns a cached instance of the VideoSummarizer."""
    return VideoSummarizer()


st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🎬 YouTube Video Summarizer")
st.markdown(
    "This tool takes a YouTube URL, transcribes the audio, and generates a summary, chapters, and key takeaways using AI."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Please create a `.env` file and add your key.")
    st.info('Example `.env` file content:\n`OPENAI_API_KEY="your-sk-key-here"`')
    st.stop()

with st.form("url_form"):
    url_input = st.text_input(
        "Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=..."
    )
    submitted = st.form_submit_button("✨ Generate Summary")

if submitted and url_input:
    # Use the cached factory function to get the summarizer instance
    summarizer = get_summarizer()
    with st.spinner(
        "Summarizing... This can take a few minutes for longer videos. Please wait."
    ):
        try:
            summary_data = summarizer.generate_summary(url_input)
            st.session_state["summary_data"] = summary_data
            st.session_state["video_url"] = url_input
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

if "summary_data" in st.session_state:
    st.success("Summary generated successfully!")

    summary_data = st.session_state["summary_data"]
    video_url = st.session_state["video_url"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Executive Summary")
        st.write(summary_data["executive_summary"])

        st.subheader("🔑 Key Takeaways")
        for takeaway in summary_data["key_takeaways"]:
            st.markdown(f"- {takeaway}")

    with col2:
        st.subheader("📖 Chapters")
        for chapter in summary_data["chapters"]:
            h, m, s = map(int, chapter["timestamp"].split(":"))
            total_seconds = h * 3600 + m * 60 + s
            chapter_url = f"{video_url}&t={total_seconds}s"
            st.markdown(
                f"**[{chapter['timestamp']}]({chapter_url})** - {chapter['title']}"
            )

    st.subheader("📥 Export Summary")
    export_col1, export_col2 = st.columns(2)

    markdown_output = to_markdown(summary_data, video_url)
    json_output = json.dumps(summary_data, indent=2, ensure_ascii=False)

    with export_col1:
        st.download_button(
            label="Download as Markdown (.md)",
            data=markdown_output,
            file_name="video_summary.md",
            mime="text/markdown",
        )
    with export_col2:
        st.download_button(
            label="Download as JSON (.json)",
            data=json_output,
            file_name="video_summary.json",
            mime="application/json",
        )
elif submitted and not url_input:
    st.warning("Please enter a YouTube URL to summarize.")

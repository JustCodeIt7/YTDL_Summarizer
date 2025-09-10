import os
import tempfile
import json
from datetime import timedelta
import streamlit as st
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import warnings

# Suppress common matmul warnings from faster_whisper on CPU
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

################################ Configuration & Model Loading ################################

# Load environment variables from a .env file
load_dotenv()


# Use Streamlit's cache to load models only once, improving performance
@st.cache_resource
def get_models():
    """Load and return the Whisper, LLM, and embedding models."""
    print("Loading models...")
    # Load the Whisper model for audio transcription
    whisper_model = WhisperModel(
        "base", device="cpu", compute_type="int8"
    )  # Use a lightweight model on CPU
    # Initialize the LLM for generation tasks
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    # Initialize the model for creating text embeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Models loaded successfully.")
    return whisper_model, llm, embedding_model


################################ Core Application Logic ################################


def download_and_transcribe(url: str, whisper_model: WhisperModel) -> list:
    """Download audio from a YouTube URL and transcribe it using Whisper."""
    # Create a temporary directory to store the downloaded audio
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure yt-dlp to download the best audio in mp3 format
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(
                temp_dir, "%(id)s.%(ext)s"
            ),  # Save file with video ID
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        # Download the audio using the specified options
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = os.path.join(temp_dir, f"{info['id']}.mp3")

        # Transcribe the downloaded audio file
        segments, _ = whisper_model.transcribe(audio_path, word_timestamps=False)
        return list(segments)


def generate_summary_and_docs(segments: list, llm: ChatOllama, url: str) -> tuple:
    """Generate a summary, chapters, and takeaways, and create document objects."""
    # Combine all transcript segments into a single string
    transcript_text = " ".join(seg.text for seg in segments)
    # Create a list of Document objects for the vectorstore
    docs = [
        Document(page_content=seg.text, metadata={"start": seg.start})
        for seg in segments
    ]

    total_duration = segments[-1].end if segments else 0
    # Dynamically determine the number of chapters based on video length
    num_chapters = min(
        max(int(total_duration / 180), 5), 12
    )  # Aim for 5-12 chapters, ~3 mins each

    # Define the prompt structure for the LLM to generate a JSON object
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert video analyst. Based on the following transcript, please generate a structured JSON object with:
        1. "executive_summary": A concise, engaging summary of the video.
        2. "chapters": A list of exactly {num_chapters} chapter objects, each with a "timestamp" (formatted as H:MM:SS from the start time) and a descriptive "title".
        3. "key_takeaways": A list of the most important bullet points or takeaways.

        TRANSCRIPT:
        "{transcript}"

        JSON OUTPUT:
        """
    )

    # Define a parser to convert the LLM's string output into JSON
    parser = JsonOutputParser()
    # Chain the prompt, LLM, and parser together
    chain = prompt_template | llm | parser

    # Format the full transcript with timestamps for better context for the LLM
    full_transcript_for_prompt = "\n".join(
        f"[{str(timedelta(seconds=int(s.start)))}] {s.text}" for s in segments
    )

    # Invoke the chain to generate the structured summary data
    summary_data = chain.invoke(
        {"transcript": full_transcript_for_prompt, "num_chapters": num_chapters}
    )
    # Add the original video URL to the metadata
    summary_data["metadata"] = {"url": url}

    return summary_data, docs


################################ Streamlit User Interface ################################

# Configure the page settings
st.set_page_config(page_title="ClipNotes", page_icon="🎬", layout="wide")
st.title("🎬 ClipNotes: YouTube Summarizer & Chat")
st.markdown(
    "Enter a YouTube URL to generate a summary and start chatting with its content."
)

# Verify the necessary API key is present in the environment
if not os.getenv("OPENAI_API_KEY"):
    st.error("API key not found. Please create a `.env` file with your key.")
    st.stop()  # Halt execution if the key is missing

# Load the AI models
whisper_model, llm, embedding_model = get_models()

# Create a form to handle the URL input and submission
with st.form("url_form"):
    url_input = st.text_input(
        "YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        value="https://youtube.com/shorts/viL-eet_s_8?si=H6rMKLbbpAL8rr00",  # Provide a default example URL
    )
    submitted = st.form_submit_button("✨ Generate & Chat")

# Process the form submission if a URL was provided
if submitted and url_input:
    st.session_state.clear()  # Reset the app state for a new video
    st.session_state.video_url = url_input
    # Execute the main processing pipeline
    try:
        # Download and transcribe the video audio
        with st.spinner("Transcribing video... (this may take a moment)"):
            segments = download_and_transcribe(url_input, whisper_model)

        # Generate summary content using the transcript
        with st.spinner("Generating summary, chapters, and takeaways..."):
            summary_data, docs = generate_summary_and_docs(segments, llm, url_input)
            st.session_state.summary_data = summary_data

        # Create and index a vectorstore for Retrieval-Augmented Generation (RAG)
        with st.spinner("Indexing video for chat..."):
            vectorstore = FAISS.from_documents(docs, embedding_model)
            # Create a question-answering chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            )
        st.success("Ready! You can now view the summary or chat with the video.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Display the full exception traceback

# Display the results if summary data exists in the session state
if "summary_data" in st.session_state:
    summary_data = st.session_state.summary_data
    video_url = st.session_state.video_url

    # --- Display Summary & Chat ---
    # Create a two-column layout for the summary and chapters
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📝 Executive Summary")
        st.write(summary_data["executive_summary"])
        st.subheader("🔑 Key Takeaways")
        # Display each key takeaway as a list item
        for takeaway in summary_data["key_takeaways"]:
            st.markdown(f"- {takeaway}")

    with col2:
        st.subheader("📖 Chapters")
        # Display each chapter with a clickable timestamp
        for chapter in summary_data["chapters"]:
            # Handle potential formatting errors in the LLM-generated timestamp
            try:
                h, m, s = map(int, chapter["timestamp"].split(":"))
                total_seconds = h * 3600 + m * 60 + s
                chapter_url = f"{video_url}&t={total_seconds}s"
                st.markdown(
                    f"**[{chapter['timestamp']}]({chapter_url})** - {chapter['title']}"
                )
            except (ValueError, KeyError):
                # Fallback for malformed chapter data
                st.markdown(f"- {chapter.get('title', 'Chapter title missing')}")

    # --- Chat Interface ---
    st.markdown("---")
    st.header("💬 Chat with the Video")

    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture and process new user input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user's message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(prompt)
                st.markdown(response["result"])
        # Add assistant's response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Handle case where the form was submitted without a URL
elif submitted:
    st.warning("Please enter a YouTube URL.")

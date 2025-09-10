# Chat with Video Feature

## Overview

The YouTube Video Summarizer now includes an interactive chat feature that allows users to ask questions about the video content after generating a summary.

## How It Works

### 1. Generate Summary First

- Enter a YouTube URL and click "Generate Summary"
- The app will download, transcribe, and analyze the video
- Summary, chapters, and key takeaways will be displayed

### 2. Chat Interface

After the summary is generated, you'll see a new "💬 Chat with Video" section with:

- **Chat History**: Previous questions and answers are displayed in expandable sections
- **Question Input**: Text field to ask questions about the video
- **Clear Chat Button**: Resets the conversation history

### 3. Ask Questions

You can ask questions like:

- "What is this video about?"
- "What are the main points discussed?"
- "Can you explain [specific topic] mentioned in the video?"
- "What did the speaker say about [topic]?"

### 4. How Answers Are Generated

- The chat feature uses the full video transcript for context
- Previous chat history is included for contextual conversations
- The AI will clearly state if information is not available in the transcript
- Responses are based only on the video content, not external knowledge

## Technical Implementation

### Backend Changes

- Added `chat_with_video()` method to `VideoSummarizer` class
- Modified `generate_summary()` to include full transcript in return data
- Uses Ollama's LLaMA 3.2 model for chat responses

### Frontend Changes

- Added chat interface in Streamlit
- Implemented session state for chat history
- Added expandable chat history display
- Included clear chat functionality

## Usage Tips

- Be specific in your questions for better answers
- The AI can only answer based on what's in the video transcript
- Use the chat history to build on previous questions
- Clear chat if you want to start a fresh conversation

## Error Handling

- If the transcript doesn't contain relevant information, the AI will indicate this
- Network or processing errors are caught and displayed to the user
- Chat functionality only appears after successful summary generation

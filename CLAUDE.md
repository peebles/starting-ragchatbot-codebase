# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Create .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended) (the server runs on port 9000)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 9000
```

### Development Server Access
- Web Interface: http://localhost:9000
- API Documentation: http://localhost:9000/docs

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system with a modular backend architecture:

### Core RAG Flow
`RAGSystem` orchestrates the entire process: User query → Tool-based AI generation → Vector search → Contextualized response

### Key Components

**FastAPI App (`app.py`)**
- Serves both API endpoints and static frontend files
- Automatic document loading on startup via `@app.on_event("startup")`
- Two main endpoints: `/api/query` (chat) and `/api/courses` (stats)

**RAG System (`rag_system.py`)**
- Main orchestrator that coordinates all components
- Manages document processing, vector storage, AI generation, and sessions
- Uses tool-based approach where AI can call search functions as needed

**Vector Storage (`vector_store.py`)**
- ChromaDB with two collections: `course_catalog` (metadata) and `course_content` (chunks)
- Persistent storage at `./chroma_db`
- Supports semantic course name resolution and lesson-specific filtering

**AI Generation (`ai_generator.py`)**
- Anthropic Claude API integration with tool calling support
- System prompt optimized for educational content with strict response guidelines
- Handles tool execution workflow with follow-up API calls

**Document Processing (`document_processor.py`)**
- Parses structured course documents with metadata extraction
- Sentence-based chunking with configurable overlap (800 chars, 100 overlap)
- Adds contextual prefixes: "Course X Lesson Y content: [chunk]"

**Search Tools (`search_tools.py`)**
- Tool-based architecture using abstract `Tool` class
- `CourseSearchTool` provides semantic search with course/lesson filtering
- `ToolManager` handles tool registration and execution

**Session Management (`session_manager.py`)**
- Maintains conversation history for context
- Configurable history length (default: 2 messages)

### Data Models (`models.py`)
- `Course`: Contains title, instructor, lessons list
- `Lesson`: Lesson number, title, optional link
- `CourseChunk`: Text content with course/lesson metadata and chunk index

### Document Format
Expected course document structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 1: [title]
Lesson Link: [url]
[lesson content...]

Lesson 2: [title]
[lesson content...]
```

### Vector Database Loading
- Automatic on server startup from `/docs` folder
- Incremental loading (skips existing courses)
- Two-stage storage: course metadata + chunked content with context enhancement

### Frontend Architecture
- Vanilla HTML/CSS/JavaScript with marked.js for markdown rendering
- Session-based conversation with loading states
- Real-time course statistics display
- Suggested questions and collapsible source attribution

## Environment Configuration
- `ANTHROPIC_API_KEY`: Required for AI generation
- `ANTHROPIC_MODEL`: Defaults to "claude-sonnet-4-20250514"
- `CHROMA_PATH`: Vector database location (default: "./chroma_db")
- `CHUNK_SIZE`: Text chunking size (default: 800)
- `CHUNK_OVERLAP`: Chunk overlap size (default: 100)
- `MAX_RESULTS`: Search result limit (default: 5)
- `MAX_HISTORY`: Conversation memory (default: 2)
- always use uv to run the server, do not use pip directly
- make sure to use uv to manage all dependecies
- use uv to run python files
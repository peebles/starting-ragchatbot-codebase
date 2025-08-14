# ChromaDB Loading Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Server Startup                               │
│                         @app.on_event("startup")                               │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Check ../docs Directory Exists                             │
│                              app.py:91                                         │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 rag_system.add_course_folder()                                 │
│                   clear_existing=False                                         │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│              Scan Files: .pdf, .docx, .txt                                     │
│              Get existing course titles for deduplication                      │
│                       rag_system.py:76-81                                      │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼ (for each new file)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DocumentProcessor.process_course_document()                 │
│                           document_processor.py:97                             │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Parse Course Metadata                                   │
│  ┌─────────────────┬─────────────────┬─────────────────┐                      │
│  │ Line 1:         │ Line 2:         │ Line 3:         │                      │
│  │ Course Title:   │ Course Link:    │ Course Instructor:│                     │
│  │ [title]         │ [url]           │ [name]          │                      │
│  └─────────────────┴─────────────────┴─────────────────┘                      │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Extract Lessons & Content                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Regex: "Lesson \d+: Title"                                            │   │
│  │  ├── Lesson 1: Introduction                                            │   │
│  │  │   └── Content text...                                               │   │
│  │  ├── Lesson 2: Advanced Topics                                         │   │
│  │  │   └── Content text...                                               │   │
│  │  └── ...                                                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Text Chunking Process                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Sentence-based chunking (800 chars, 100 overlap)                       │   │
│  │ ├── Chunk 0: "Course Title Lesson N content: [text...]"                │   │
│  │ ├── Chunk 1: "Course Title Lesson N content: [text...]"                │   │
│  │ └── ...                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ChromaDB Storage                                      │
│                            vector_store.py                                     │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ChromaDB Persistent Client                                  │
│                         Path: ./chroma_db                                      │
│                    Embedding: all-MiniLM-L6-v2                                 │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Two Collections                                       │
│                                                                                 │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐   │
│  │      course_catalog         │    │         course_content             │   │
│  │                             │    │                                     │   │
│  │ Documents:                  │    │ Documents:                          │   │
│  │ • Course titles             │    │ • Chunked text with context        │   │
│  │                             │    │   "Course X Lesson Y content: ..." │   │
│  │ Metadata:                   │    │                                     │   │
│  │ • Instructor                │    │ Metadata:                           │   │
│  │ • Course links             │    │ • Course title                      │   │
│  │ • Lessons JSON             │    │ • Lesson number                     │   │
│  │ • Lesson count             │    │ • Chunk index                       │   │
│  │                             │    │                                     │   │
│  │ IDs:                        │    │ IDs:                                │   │
│  │ • Course titles             │    │ • "{course_title}_{chunk_index}"   │   │
│  └─────────────────────────────┘    └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Final State                                       │
│                                                                                 │
│  Vector Database Ready for:                                                    │
│  ├── Semantic search across course content                                     │
│  ├── Course name resolution                                                    │
│  ├── Lesson-specific filtering                                                 │
│  ├── Source attribution                                                        │
│  └── Analytics (course count, titles)                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Flow Characteristics

**Timing**: Automatic on FastAPI server startup  
**Persistence**: Data survives server restarts  
**Deduplication**: Only processes new courses  
**Error Handling**: Continues if individual files fail  
**Context Enhancement**: Chunks prefixed with course/lesson info  
**Dual Storage**: Metadata separate from content for efficient search
import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
        ]
    )

@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Test Course Lesson 1 content: This is the introduction to the test course.",
            course_title="Test Course", 
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Test Course Lesson 1 content: We'll cover basic concepts and terminology.",
            course_title="Test Course",
            lesson_number=1, 
            chunk_index=1
        ),
        CourseChunk(
            content="Course Test Course Lesson 2 content: Advanced topics include complex algorithms.",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2
        )
    ]

@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_vector_store(temp_chroma_path):
    """Create a test vector store with sample data"""
    # Use default embedding for testing
    store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
    return store

@pytest.fixture
def populated_vector_store(test_vector_store, sample_course, sample_chunks):
    """Create a vector store populated with test data"""
    test_vector_store.add_course_metadata(sample_course)
    test_vector_store.add_course_content(sample_chunks)
    return test_vector_store

@pytest.fixture
def course_search_tool(populated_vector_store):
    """Create a CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)

@pytest.fixture
def course_outline_tool(populated_vector_store):
    """Create a CourseOutlineTool with populated data"""
    return CourseOutlineTool(populated_vector_store)

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def test_config():
    """Create test configuration"""
    return Config()

@pytest.fixture
def mock_ai_generator(mock_anthropic_client, test_config):
    """Create AIGenerator with mocked Anthropic client"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        return AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

# Helper functions for testing
def create_search_results(documents: List[str], metadata: List[Dict[str, Any]] = None) -> SearchResults:
    """Helper to create SearchResults for testing"""
    if metadata is None:
        metadata = [{"course_title": "Test Course", "lesson_number": 1, "chunk_index": i} for i in range(len(documents))]
    
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=[0.1] * len(documents)
    )
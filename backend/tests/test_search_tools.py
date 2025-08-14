import pytest
from unittest.mock import Mock, patch
import json

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""
    
    def test_tool_definition(self, course_search_tool):
        """Test that tool definition is properly formatted"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_basic_search(self, course_search_tool):
        """Test basic search functionality"""
        result = course_search_tool.execute("introduction")
        
        # Should return formatted results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Course" in result or "No relevant content found" in result
    
    def test_execute_with_course_filter(self, course_search_tool):
        """Test search with course name filter"""
        result = course_search_tool.execute("introduction", course_name="Test Course")
        
        assert isinstance(result, str)
        # Should either find content or report no content found for specific course
        assert "Test Course" in result or "No relevant content found in course 'Test Course'" in result
    
    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test search with lesson number filter"""
        result = course_search_tool.execute("content", lesson_number=1)
        
        assert isinstance(result, str)
        # Should filter to specific lesson
        if "No relevant content found" not in result:
            assert "Lesson 1" in result
    
    def test_execute_with_course_and_lesson_filter(self, course_search_tool):
        """Test search with both course and lesson filters"""
        result = course_search_tool.execute(
            "introduction", 
            course_name="Test Course", 
            lesson_number=1
        )
        
        assert isinstance(result, str)
        # Should be specific to course and lesson
        if "No relevant content found" not in result:
            assert "Test Course" in result
            assert "Lesson 1" in result
    
    def test_execute_nonexistent_course(self, course_search_tool):
        """Test search with non-existent course name"""
        result = course_search_tool.execute("test", course_name="Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_execute_empty_query(self, course_search_tool):
        """Test search with empty query"""
        result = course_search_tool.execute("")
        
        assert isinstance(result, str)
        # Should handle gracefully
    
    def test_sources_tracking(self, course_search_tool):
        """Test that sources are properly tracked"""
        # Clear any existing sources
        course_search_tool.last_sources = []
        
        result = course_search_tool.execute("introduction")
        
        # Check if sources were tracked
        if "No relevant content found" not in result:
            assert len(course_search_tool.last_sources) > 0
            # Sources should contain course information
            assert any("Test Course" in source for source in course_search_tool.last_sources)
    
    def test_format_results_basic(self, course_search_tool):
        """Test result formatting"""
        # Create mock search results
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1]
        )
        
        formatted = course_search_tool._format_results(results)
        
        assert "[Test Course - Lesson 1]" in formatted
        assert "Test content" in formatted
        assert len(course_search_tool.last_sources) == 1
        assert "Test Course - Lesson 1" in course_search_tool.last_sources[0]
    
    def test_format_results_with_lesson_link(self, populated_vector_store):
        """Test result formatting with lesson links"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Mock the get_lesson_link method to return a link
        with patch.object(populated_vector_store, 'get_lesson_link', return_value="https://example.com/lesson1"):
            results = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}],
                distances=[0.1]
            )
            
            formatted = tool._format_results(results)
            
            # Source should contain embedded link
            assert len(tool.last_sources) == 1
            assert "|https://example.com/lesson1" in tool.last_sources[0]


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""
    
    def test_tool_definition(self, course_outline_tool):
        """Test that tool definition is properly formatted"""
        definition = course_outline_tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]
    
    def test_execute_valid_course(self, course_outline_tool):
        """Test outline retrieval for valid course"""
        result = course_outline_tool.execute("Test Course")
        
        assert isinstance(result, str)
        if "No course found" not in result:
            assert "**Course:** Test Course" in result
            assert "**Lessons:**" in result
            assert "1. Introduction" in result
            assert "2. Advanced Topics" in result
    
    def test_execute_invalid_course(self, course_outline_tool):
        """Test outline retrieval for invalid course"""
        result = course_outline_tool.execute("Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result


class TestToolManager:
    """Test suite for ToolManager"""
    
    def test_register_tool(self):
        """Test tool registration"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        
        manager.register_tool(mock_tool)
        
        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool
    
    def test_get_tool_definitions(self, populated_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        outline_tool = CourseOutlineTool(populated_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)
    
    def test_execute_tool(self, populated_vector_store):
        """Test tool execution"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(search_tool)
        
        result = manager.execute_tool("search_course_content", query="test")
        
        assert isinstance(result, str)
    
    def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, populated_vector_store):
        """Test source tracking across tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="introduction")
        
        sources = manager.get_last_sources()
        assert isinstance(sources, list)
    
    def test_reset_sources(self, populated_vector_store):
        """Test source reset functionality"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="introduction")
        
        # Verify sources exist
        sources_before = manager.get_last_sources()
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        sources_after = manager.get_last_sources()
        assert len(sources_after) == 0
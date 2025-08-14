import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore
from search_tools import CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Integration tests for RAG System"""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration with temporary paths"""
        config = Config()
        config.ANTHROPIC_API_KEY = "test_key"
        config.CHROMA_PATH = tempfile.mkdtemp()
        return config
    
    @pytest.fixture
    def mock_rag_system(self, test_config):
        """Create RAG system with mocked AI generator"""
        with patch('rag_system.AIGenerator') as mock_ai_gen:
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Test AI response"
            mock_ai_gen.return_value = mock_ai_instance
            
            rag = RAGSystem(test_config)
            rag.mock_ai_instance = mock_ai_instance  # Store for test access
            return rag
    
    def test_initialization(self, test_config):
        """Test RAG system initialization"""
        with patch('rag_system.AIGenerator'):
            rag = RAGSystem(test_config)
            
            assert rag.config == test_config
            assert rag.document_processor is not None
            assert rag.vector_store is not None
            assert rag.ai_generator is not None
            assert rag.session_manager is not None
            assert rag.tool_manager is not None
            
            # Verify tools are registered
            tool_definitions = rag.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 2
            assert any(t["name"] == "search_course_content" for t in tool_definitions)
            assert any(t["name"] == "get_course_outline" for t in tool_definitions)
    
    def test_add_course_document_success(self, mock_rag_system, sample_course, sample_chunks):
        """Test successful course document addition"""
        # Mock document processor method
        with patch.object(mock_rag_system.document_processor, 'process_course_document', return_value=(sample_course, sample_chunks)):
            # Mock vector store methods
            with patch.object(mock_rag_system.vector_store, 'add_course_metadata') as mock_add_meta, \
                 patch.object(mock_rag_system.vector_store, 'add_course_content') as mock_add_content:
                
                course, chunk_count = mock_rag_system.add_course_document("test_file.txt")
                
                assert course == sample_course
                assert chunk_count == len(sample_chunks)
                
                # Verify vector store was called
                mock_add_meta.assert_called_once_with(sample_course)
                mock_add_content.assert_called_once_with(sample_chunks)
    
    def test_add_course_document_failure(self, mock_rag_system):
        """Test course document addition failure handling"""
        # Mock document processor to raise exception
        with patch.object(mock_rag_system.document_processor, 'process_course_document', side_effect=Exception("Test error")):
            course, chunk_count = mock_rag_system.add_course_document("test_file.txt")
            
            assert course is None
            assert chunk_count == 0
    
    def test_query_without_session(self, mock_rag_system):
        """Test query processing without session"""
        response, sources = mock_rag_system.query("What is AI?")
        
        assert response == "Test AI response"
        assert isinstance(sources, list)
        
        # Verify AI generator was called correctly
        mock_rag_system.mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_rag_system.mock_ai_instance.generate_response.call_args
        
        assert "Answer this question about course materials: What is AI?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    def test_query_with_session(self, mock_rag_system):
        """Test query processing with session"""
        # Create session and add some history
        session_id = mock_rag_system.session_manager.create_session()
        mock_rag_system.session_manager.add_exchange(session_id, "Previous question", "Previous answer")
        
        response, sources = mock_rag_system.query("What is AI?", session_id=session_id)
        
        assert response == "Test AI response"
        
        # Verify conversation history was passed
        call_args = mock_rag_system.mock_ai_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] is not None
        assert "Previous question" in call_args[1]["conversation_history"]
    
    def test_query_with_tool_execution(self, mock_rag_system):
        """Test query that triggers tool execution"""
        # Mock tool manager methods
        with patch.object(mock_rag_system.tool_manager, 'get_last_sources', return_value=["Test Course - Lesson 1"]) as mock_get_sources, \
             patch.object(mock_rag_system.tool_manager, 'reset_sources') as mock_reset_sources:
            
            response, sources = mock_rag_system.query("What is machine learning?")
            
            assert response == "Test AI response"
            assert sources == ["Test Course - Lesson 1"]
            
            # Verify sources were retrieved and reset
            mock_get_sources.assert_called_once()
            mock_reset_sources.assert_called_once()
    
    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval"""
        # Mock vector store analytics methods
        with patch.object(mock_rag_system.vector_store, 'get_course_count', return_value=3) as mock_count, \
             patch.object(mock_rag_system.vector_store, 'get_existing_course_titles', return_value=["Course 1", "Course 2", "Course 3"]) as mock_titles:
            
            analytics = mock_rag_system.get_course_analytics()
            
            assert analytics["total_courses"] == 3
            assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
    
    @patch('os.path.isfile')
    @patch('os.path.exists') 
    @patch('os.listdir')
    def test_add_course_folder_with_files(self, mock_listdir, mock_exists, mock_isfile, mock_rag_system):
        """Test adding course folder with multiple files"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "not_a_course.log"]
        # Mock isfile to return True only for valid course files
        def mock_isfile_side_effect(path):
            return path.endswith(('.txt', '.pdf'))
        mock_isfile.side_effect = mock_isfile_side_effect
        
        # Mock existing courses and processing with different courses for each file
        def mock_process_side_effect(file_path):
            if "course1.txt" in file_path:
                return Course(title="Course 1", instructor="Instructor 1"), [CourseChunk(content="test1", course_title="Course 1", chunk_index=0)]
            elif "course2.pdf" in file_path:
                return Course(title="Course 2", instructor="Instructor 2"), [CourseChunk(content="test2", course_title="Course 2", chunk_index=0)]
            else:
                raise Exception("Unsupported file")
        
        with patch.object(mock_rag_system.vector_store, 'get_existing_course_titles', return_value=[]) as mock_titles, \
             patch.object(mock_rag_system.document_processor, 'process_course_document', side_effect=mock_process_side_effect) as mock_process, \
             patch.object(mock_rag_system.vector_store, 'add_course_metadata') as mock_add_meta, \
             patch.object(mock_rag_system.vector_store, 'add_course_content') as mock_add_content:
            
            courses, chunks = mock_rag_system.add_course_folder("/test/folder")
            
            # Should process 2 valid files (txt and pdf)
            assert courses == 2
            assert chunks == 2
            
            # Verify vector store was called for each course
            assert mock_add_meta.call_count == 2
            assert mock_add_content.call_count == 2
    
    def test_add_course_folder_nonexistent(self, mock_rag_system):
        """Test adding course folder that doesn't exist"""
        courses, chunks = mock_rag_system.add_course_folder("/nonexistent/folder")
        
        assert courses == 0
        assert chunks == 0


class TestRAGSystemWithRealVectorStore:
    """Integration tests with real vector store"""
    
    @pytest.fixture
    def real_rag_system(self, temp_chroma_path):
        """Create RAG system with real vector store but mocked AI"""
        config = Config()
        config.ANTHROPIC_API_KEY = "test_key"
        config.CHROMA_PATH = temp_chroma_path
        
        with patch('rag_system.AIGenerator') as mock_ai_gen:
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Test response from real vector store"
            mock_ai_gen.return_value = mock_ai_instance
            
            rag = RAGSystem(config)
            rag.mock_ai_instance = mock_ai_instance
            return rag
    
    def test_full_workflow_with_real_data(self, real_rag_system, sample_course, sample_chunks):
        """Test full workflow with real vector store and data"""
        # Add test data
        real_rag_system.vector_store.add_course_metadata(sample_course)
        real_rag_system.vector_store.add_course_content(sample_chunks)
        
        # Execute query
        response, sources = real_rag_system.query("What is introduction?")
        
        assert response == "Test response from real vector store"
        
        # Verify AI was called with proper tools
        call_args = real_rag_system.mock_ai_instance.generate_response.call_args
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
        
        # Verify tools are functional
        tool_defs = call_args[1]["tools"]
        assert len(tool_defs) == 2
    
    def test_real_tool_execution(self, real_rag_system, sample_course, sample_chunks):
        """Test that tools actually work with real vector store"""
        # Add test data
        real_rag_system.vector_store.add_course_metadata(sample_course)
        real_rag_system.vector_store.add_course_content(sample_chunks)
        
        # Test search tool directly
        search_result = real_rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="introduction"
        )
        
        assert isinstance(search_result, str)
        assert len(search_result) > 0
        
        # Test outline tool directly
        outline_result = real_rag_system.tool_manager.execute_tool(
            "get_course_outline", 
            course_name="Test Course"
        )
        
        assert isinstance(outline_result, str)
        assert "Test Course" in outline_result or "No course found" in outline_result
import pytest
import os
import sys
from unittest.mock import patch, Mock
import chromadb
from config import Config

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore
from document_processor import DocumentProcessor
from rag_system import RAGSystem


class TestSystemDiagnostics:
    """Diagnostic tests to verify current system state"""
    
    def test_anthropic_api_key_availability(self):
        """Test if Anthropic API key is configured"""
        config = Config()
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY not set in environment"
        assert len(config.ANTHROPIC_API_KEY) > 10, "ANTHROPIC_API_KEY appears to be invalid (too short)"
    
    def test_chroma_db_exists_and_accessible(self):
        """Test if ChromaDB directory exists and is accessible"""
        config = Config()
        chroma_path = config.CHROMA_PATH
        
        assert os.path.exists(chroma_path), f"ChromaDB path {chroma_path} does not exist"
        assert os.path.isdir(chroma_path), f"ChromaDB path {chroma_path} is not a directory"
        assert os.access(chroma_path, os.R_OK), f"ChromaDB path {chroma_path} is not readable"
        assert os.access(chroma_path, os.W_OK), f"ChromaDB path {chroma_path} is not writable"
    
    def test_chroma_collections_exist(self):
        """Test if ChromaDB collections exist and have data"""
        config = Config()
        
        try:
            # Try to connect to existing ChromaDB
            client = chromadb.PersistentClient(
                path=config.CHROMA_PATH,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            # Check if collections exist
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            assert "course_catalog" in collection_names, "course_catalog collection not found"
            assert "course_content" in collection_names, "course_content collection not found"
            
            # Check if collections have data
            catalog_collection = client.get_collection("course_catalog")
            content_collection = client.get_collection("course_content")
            
            catalog_count = catalog_collection.count()
            content_count = content_collection.count()
            
            print(f"Course catalog entries: {catalog_count}")
            print(f"Course content entries: {content_count}")
            
            assert catalog_count > 0, "course_catalog collection is empty"
            assert content_count > 0, "course_content collection is empty"
            
        except Exception as e:
            pytest.fail(f"Failed to access ChromaDB: {e}")
    
    def test_embedding_model_functionality(self):
        """Test if embedding model is working"""
        config = Config()
        
        try:
            # Test with default embedding fallback
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=1)
            
            # Try to perform a simple search
            results = store.search("test query", limit=1)
            
            # Should not error, even if empty
            assert results is not None
            assert hasattr(results, 'documents')
            assert hasattr(results, 'metadata')
            
        except Exception as e:
            pytest.fail(f"Embedding model test failed: {e}")
    
    @patch('anthropic.Anthropic')
    def test_anthropic_api_connectivity(self, mock_anthropic):
        """Test Anthropic API connectivity and response format"""
        config = Config()
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        from ai_generator import AIGenerator
        
        generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Test basic API call
        response = generator.generate_response("Test query")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()
    
    def test_document_files_exist(self):
        """Test if course document files exist"""
        docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
        
        assert os.path.exists(docs_path), f"Documents directory {docs_path} does not exist"
        
        files = [f for f in os.listdir(docs_path) if f.endswith(('.txt', '.pdf', '.docx'))]
        assert len(files) > 0, f"No course document files found in {docs_path}"
        
        print(f"Found {len(files)} course document files: {files}")
    
    def test_document_processing_functionality(self):
        """Test if document processor can parse existing files"""
        config = Config()
        processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
        
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
            
            if files:
                test_file = os.path.join(docs_path, files[0])
                
                try:
                    course, chunks = processor.process_course_document(test_file)
                    
                    assert course is not None, f"Failed to parse course from {test_file}"
                    assert course.title, "Course title is empty"
                    assert len(chunks) > 0, f"No chunks created from {test_file}"
                    
                    print(f"Successfully processed {test_file}: {course.title} with {len(chunks)} chunks")
                    
                except Exception as e:
                    pytest.fail(f"Document processing failed for {test_file}: {e}")


class TestCurrentSystemState:
    """Tests that examine the current running system state"""
    
    def test_vector_store_search_functionality(self):
        """Test if vector store search is working with current data"""
        config = Config()
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=5)
        
        # Test basic search
        results = store.search("introduction")
        
        print(f"Search results for 'introduction': {len(results.documents)} documents")
        if results.error:
            print(f"Search error: {results.error}")
        
        if not results.is_empty():
            print(f"First result: {results.documents[0][:100]}...")
            print(f"First metadata: {results.metadata[0]}")
        
        # Test should not fail, but results might be empty
        assert results is not None
        assert isinstance(results.documents, list)
        assert isinstance(results.metadata, list)
    
    def test_course_name_resolution(self):
        """Test if course name resolution is working"""
        config = Config()
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=5)
        
        # Test course name resolution
        course_titles = store.get_existing_course_titles()
        print(f"Existing course titles: {course_titles}")
        
        if course_titles:
            # Test resolving first course
            first_course = course_titles[0]
            resolved = store._resolve_course_name(first_course)
            print(f"Resolution test - Input: {first_course}, Resolved: {resolved}")
            
            # Should resolve to itself
            assert resolved == first_course or resolved is None
            
            # Test partial match
            partial_name = first_course.split()[0] if ' ' in first_course else first_course[:5]
            partial_resolved = store._resolve_course_name(partial_name)
            print(f"Partial resolution test - Input: {partial_name}, Resolved: {partial_resolved}")
    
    def test_tool_manager_with_current_data(self):
        """Test tool manager with current system data"""
        config = Config()
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=5)
        
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
        
        manager = ToolManager()
        search_tool = CourseSearchTool(store)
        outline_tool = CourseOutlineTool(store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Test search tool execution
        search_result = manager.execute_tool("search_course_content", query="introduction")
        print(f"Direct tool search result: {search_result}")
        
        # Test outline tool execution
        course_titles = store.get_existing_course_titles()
        if course_titles:
            outline_result = manager.execute_tool("get_course_outline", course_name=course_titles[0])
            print(f"Direct tool outline result: {outline_result}")
        
        assert isinstance(search_result, str)
        assert len(search_result) > 0


class TestAnthropicAPIIntegration:
    """Test actual Anthropic API integration (requires valid API key)"""
    
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
    def test_real_anthropic_api_call(self):
        """Test actual API call to Anthropic (requires valid API key)"""
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "":
            pytest.skip("ANTHROPIC_API_KEY not configured")
        
        from ai_generator import AIGenerator
        
        try:
            generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            response = generator.generate_response("What is 2+2?")
            
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"API response test successful: {response}")
            
        except Exception as e:
            pytest.fail(f"Anthropic API call failed: {e}")
    
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")  
    def test_anthropic_tool_calling_format(self):
        """Test that tool definitions are accepted by Anthropic API"""
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "":
            pytest.skip("ANTHROPIC_API_KEY not configured")
        
        from ai_generator import AIGenerator
        from search_tools import CourseSearchTool, ToolManager
        
        # Create minimal vector store for tool definition
        temp_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=1)
        search_tool = CourseSearchTool(temp_store)
        
        tool_definitions = [search_tool.get_tool_definition()]
        
        try:
            generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            
            # Test API call with tools (should not error on tool format)
            response = generator.generate_response(
                "What is 2+2?", 
                tools=tool_definitions,
                tool_manager=None  # Won't execute tools, just test format
            )
            
            assert isinstance(response, str)
            print(f"Tool format test successful")
            
        except Exception as e:
            pytest.fail(f"Tool definition format rejected by API: {e}")
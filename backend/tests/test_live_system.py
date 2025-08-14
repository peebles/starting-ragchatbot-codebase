import pytest
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem


class TestLiveSystem:
    """Test the actual running system with real data"""
    
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
    def test_real_rag_query(self):
        """Test a real query against the current system"""
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "":
            pytest.skip("ANTHROPIC_API_KEY not configured")
        
        try:
            # Create real RAG system
            rag = RAGSystem(config)
            
            # Test basic query
            response, sources = rag.query("What is MCP?")
            
            print(f"Query response: {response}")
            print(f"Sources: {sources}")
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "query failed" not in response.lower()
            
        except Exception as e:
            pytest.fail(f"Real RAG query failed: {e}")
    
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key available")
    def test_real_rag_query_with_course_specific(self):
        """Test a course-specific query"""
        config = Config()
        
        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "":
            pytest.skip("ANTHROPIC_API_KEY not configured")
        
        try:
            rag = RAGSystem(config)
            
            # Test course-specific query
            response, sources = rag.query("What is computer use with Anthropic?")
            
            print(f"Course-specific query response: {response}")
            print(f"Sources: {sources}")
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "query failed" not in response.lower()
            
        except Exception as e:
            pytest.fail(f"Course-specific RAG query failed: {e}")
    
    def test_tool_execution_directly(self):
        """Test direct tool execution with current system"""
        config = Config()
        
        try:
            rag = RAGSystem(config)
            
            # Test search tool directly
            search_result = rag.tool_manager.execute_tool(
                "search_course_content", 
                query="What is MCP?"
            )
            
            print(f"Direct search tool result: {search_result}")
            
            # Test outline tool directly  
            outline_result = rag.tool_manager.execute_tool(
                "get_course_outline",
                course_name="MCP"
            )
            
            print(f"Direct outline tool result: {outline_result}")
            
            assert isinstance(search_result, str)
            assert isinstance(outline_result, str)
            assert len(search_result) > 0
            assert len(outline_result) > 0
            
        except Exception as e:
            pytest.fail(f"Direct tool execution failed: {e}")
    
    def test_vector_store_direct_search(self):
        """Test vector store search directly"""
        config = Config()
        
        try:
            from vector_store import VectorStore
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, max_results=5)
            
            # Test basic search
            results = store.search("MCP")
            
            print(f"Vector store search results: {len(results.documents)} documents")
            if results.error:
                print(f"Error: {results.error}")
            else:
                print(f"First result: {results.documents[0][:200]}..." if results.documents else "No documents")
                print(f"First metadata: {results.metadata[0]}" if results.metadata else "No metadata")
            
            assert results is not None
            assert not results.error, f"Vector store search error: {results.error}"
            
        except Exception as e:
            pytest.fail(f"Vector store direct search failed: {e}")
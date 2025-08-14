import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool


class TestAIGenerator:
    """Test suite for AIGenerator"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test_key", "test_model")
            
            mock_anthropic.assert_called_once_with(api_key="test_key")
            assert generator.model == "test_model"
            assert generator.base_params["model"] == "test_model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, mock_ai_generator):
        """Test basic response generation without tools"""
        result = mock_ai_generator.generate_response("What is AI?")
        
        assert result == "Test response"
        
        # Verify API call was made correctly
        mock_ai_generator.client.messages.create.assert_called_once()
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        
        assert call_args["model"] == mock_ai_generator.model
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert call_args["system"] == mock_ai_generator.SYSTEM_PROMPT
    
    def test_generate_response_with_conversation_history(self, mock_ai_generator):
        """Test response generation with conversation history"""
        history = "User: Hello\nAssistant: Hi there!"
        
        result = mock_ai_generator.generate_response("What is AI?", conversation_history=history)
        
        assert result == "Test response"
        
        # Verify system prompt includes history
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        assert history in call_args["system"]
    
    def test_generate_response_with_tools_no_tool_use(self, mock_ai_generator):
        """Test response generation with tools available but not used"""
        tools = [{"name": "test_tool", "description": "Test tool"}]
        tool_manager = Mock()
        
        result = mock_ai_generator.generate_response(
            "What is AI?", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Test response"
        
        # Verify tools were passed to API
        call_args = mock_ai_generator.client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, mock_ai_generator):
        """Test response generation when AI decides to use tools"""
        # Mock initial response that indicates tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test query"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_content]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response with tool results"
        
        # Set up client to return initial response, then final response
        mock_ai_generator.client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool execution result"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = mock_ai_generator.generate_response(
            "What is test content?",
            tools=tools,
            tool_manager=tool_manager
        )
        
        assert result == "Final response with tool results"
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert mock_ai_generator.client.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, mock_ai_generator):
        """Test tool execution workflow with single tool call"""
        # Mock tool use content
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results here"
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt"
        }
        
        # Mock final API response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response"
        mock_ai_generator.client.messages.create.return_value = mock_final_response
        
        result = mock_ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            tool_manager
        )
        
        assert result == "Final response"
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test"
        )
    
    def test_handle_tool_execution_multiple_tools(self, mock_ai_generator):
        """Test tool execution workflow with multiple tool calls"""
        # Mock multiple tool use contents
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search_course_content"
        mock_tool1.id = "tool_123"
        mock_tool1.input = {"query": "test1"}
        
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use" 
        mock_tool2.name = "get_course_outline"
        mock_tool2.id = "tool_456"
        mock_tool2.input = {"course_name": "Test Course"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool1, mock_tool2]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt"
        }
        
        # Mock final API response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response"
        mock_ai_generator.client.messages.create.return_value = mock_final_response
        
        result = mock_ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            tool_manager
        )
        
        assert result == "Final response"
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="test1")
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Test Course")


class TestToolManagerIntegration:
    """Test ToolManager with actual tool implementations"""
    
    def test_real_tool_registration_and_execution(self, populated_vector_store):
        """Test real tool registration and execution"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        outline_tool = CourseOutlineTool(populated_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Test search tool execution
        search_result = manager.execute_tool("search_course_content", query="introduction")
        assert isinstance(search_result, str)
        
        # Test outline tool execution  
        outline_result = manager.execute_tool("get_course_outline", course_name="Test Course")
        assert isinstance(outline_result, str)
        
        # Test source tracking
        sources = manager.get_last_sources()
        assert isinstance(sources, list)
    
    def test_tool_definitions_format_for_anthropic(self, populated_vector_store):
        """Test that tool definitions are properly formatted for Anthropic API"""
        manager = ToolManager()
        search_tool = CourseSearchTool(populated_vector_store)
        outline_tool = CourseOutlineTool(populated_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        # Verify structure matches Anthropic's expected format
        assert len(definitions) == 2
        
        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition
            assert definition["input_schema"]["type"] == "object"
            assert "properties" in definition["input_schema"]
            assert "required" in definition["input_schema"]
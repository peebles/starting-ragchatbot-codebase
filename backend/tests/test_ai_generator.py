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
        mock_final_response.stop_reason = "end_turn"
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
    
    def test_handle_sequential_tool_execution_single_round(self, mock_ai_generator):
        """Test sequential tool execution with single round"""
        # Mock tool use content
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_content]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results here"
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content", "description": "Search tool"}]
        }
        
        # Mock final API response (no more tool use)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response"
        mock_ai_generator.client.messages.create.return_value = mock_final_response
        
        result = mock_ai_generator._handle_sequential_tool_execution(
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
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool1, mock_tool2]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content", "description": "Search tool"}]
        }
        
        # Mock final API response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response"
        mock_ai_generator.client.messages.create.return_value = mock_final_response
        
        result = mock_ai_generator._handle_sequential_tool_execution(
            mock_initial_response,
            base_params,
            tool_manager
        )
        
        assert result == "Final response"
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="test1")
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Test Course")
    
    def test_sequential_tool_execution_two_rounds(self, mock_ai_generator):
        """Test two rounds of sequential tool execution"""
        # Round 1: AI uses outline tool
        round1_tool = Mock()
        round1_tool.type = "tool_use"
        round1_tool.name = "get_course_outline"
        round1_tool.id = "tool_1"
        round1_tool.input = {"course_name": "Test Course"}
        
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [round1_tool]
        
        # Round 2: AI uses search tool based on outline results
        round2_tool = Mock()
        round2_tool.type = "tool_use"
        round2_tool.name = "search_course_content"
        round2_tool.id = "tool_2"
        round2_tool.input = {"query": "lesson 1", "course_name": "Test Course"}
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [round2_tool]
        
        # Final response after 2 rounds
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Sequential tool execution complete"
        final_response.content = [final_content]
        
        # Set up API call sequence
        mock_ai_generator.client.messages.create.side_effect = [
            round2_response,  # After round 1 tools (round1_response is passed in as initial_response)
            final_response    # After round 2 tools
        ]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Course outline results",  # Round 1
            "Search results"           # Round 2
        ]
        
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Test Course lesson 1"}],
            "system": "Test system prompt",
            "tools": [
                {"name": "get_course_outline", "description": "Get outline"},
                {"name": "search_course_content", "description": "Search content"}
            ]
        }
        
        result = mock_ai_generator._handle_sequential_tool_execution(
            round1_response,
            base_params,
            tool_manager
        )
        
        assert result == "Sequential tool execution complete"
        assert tool_manager.execute_tool.call_count == 2
        assert mock_ai_generator.client.messages.create.call_count == 2
    
    def test_max_rounds_exceeded(self, mock_ai_generator):
        """Test behavior when max tool rounds is exceeded"""
        # Create response that always requests tools
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_1"
        tool_content.input = {"query": "test"}
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_content]
        
        # Set low limit for testing
        mock_ai_generator.max_tool_rounds = 1
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        # Mock final response without tool use (for _extract_response_text)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Tool execution stopped after max rounds"
        final_response.content = [final_content]
        
        # Mock API to return final response after 1 round of tools
        mock_ai_generator.client.messages.create.return_value = final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content", "description": "Search"}]
        }
        
        result = mock_ai_generator._handle_sequential_tool_execution(
            tool_response,
            base_params,
            tool_manager
        )
        
        # Should return the final response text
        assert result == "Tool execution stopped after max rounds"
        # Verify only 1 round was executed due to max_tool_rounds = 1
        assert tool_manager.execute_tool.call_count == 1
    
    def test_max_rounds_exceeded_forces_final_text_response(self, mock_ai_generator):
        """Test that when max rounds is exceeded, a final call without tools is made to get text response"""
        # Create tool use response that Claude keeps wanting to use
        tool_content = Mock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_1"
        tool_content.input = {"query": "RAG"}
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_content]
        
        # Set max_tool_rounds = 1 so we hit limit quickly
        mock_ai_generator.max_tool_rounds = 1
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result about RAG"
        
        # Mock final response after removing tools
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Based on the search results, RAG is a technique..."
        final_response.content = [final_content]
        
        # Mock API calls: first returns another tool_use response, then final text response
        mock_ai_generator.client.messages.create.side_effect = [
            tool_response,  # After round 1 tools, Claude still wants more tools
            final_response  # Final call without tools forces text response
        ]
        
        base_params = {
            "messages": [{"role": "user", "content": "Are there any courses that explain what RAG is?"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content", "description": "Search"}]
        }
        
        result = mock_ai_generator._handle_sequential_tool_execution(
            tool_response,  # Initial response wants tools
            base_params,
            tool_manager
        )
        
        # Should get final text response, not "No text response available"
        assert result == "Based on the search results, RAG is a technique..."
        
        # Verify the sequence: 1 tool execution, then 2 API calls
        assert tool_manager.execute_tool.call_count == 1
        assert mock_ai_generator.client.messages.create.call_count == 2
        
        # Verify final API call was made without tools
        final_call_args = mock_ai_generator.client.messages.create.call_args_list[1][1]
        assert "tools" not in final_call_args  # No tools parameter in final call
    
    def test_course_name_resolution_scenario(self, mock_ai_generator):
        """Test the specific scenario: abbreviated course name -> outline tool -> search tool -> final response"""
        # Round 1: Claude uses outline tool to resolve "MCP"
        outline_tool = Mock()
        outline_tool.type = "tool_use"
        outline_tool.name = "get_course_outline"
        outline_tool.id = "tool_1"
        outline_tool.input = {"course_name": "MCP"}
        
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [outline_tool]
        
        # Round 2: Claude uses search tool with resolved course name  
        search_tool = Mock()
        search_tool.type = "tool_use"
        search_tool.name = "search_course_content"
        search_tool.id = "tool_2"
        search_tool.input = {"query": "lesson 5", "course_name": "MCP: Build Rich-Context AI Apps with Anthropic"}
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [search_tool]
        
        # Final response: Claude provides the answer based on search results
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_content = Mock()
        final_content.text = "Lesson 5 of the MCP course covers server configuration and integration..."
        final_response.content = [final_content]
        
        # Set up API call sequence: round1 -> search tool call -> final without tools
        mock_ai_generator.client.messages.create.side_effect = [
            round2_response,  # After outline tool execution
            round2_response,  # After search tool execution (still wants more tools)  
            final_response    # Final call without tools
        ]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Course: MCP: Build Rich-Context AI Apps with Anthropic\nLessons: 1, 2, 3, 4, 5...",  # Outline result
            "Lesson 5 content about server configuration..."  # Search result
        ]
        
        base_params = {
            "messages": [{"role": "user", "content": "What was covered in lesson 5 of the MCP course?"}],
            "system": "Test system prompt", 
            "tools": [
                {"name": "get_course_outline", "description": "Get outline"},
                {"name": "search_course_content", "description": "Search content"}
            ]
        }
        
        result = mock_ai_generator._handle_sequential_tool_execution(
            round1_response,  # Start with outline tool call
            base_params,
            tool_manager
        )
        
        # Should get final response about lesson 5, not "No response generated"
        assert result == "Lesson 5 of the MCP course covers server configuration and integration..."
        
        # Verify the full sequence: outline tool + search tool
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 5", course_name="MCP: Build Rich-Context AI Apps with Anthropic")
        
        # Verify API calls: 2 rounds + 1 final without tools = 3 total
        assert mock_ai_generator.client.messages.create.call_count == 3


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
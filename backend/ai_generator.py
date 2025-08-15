import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage:
- **Content search tool**: Use for questions about specific course content or detailed educational materials
- **Course outline tool**: Use for questions about course structure, lesson lists, or course overviews
- **Sequential tool calls**: You can use multiple tools across multiple rounds to gather comprehensive information
- **Tool call strategy**: Use tools iteratively to build complete answers (e.g., first get course outline, then search specific content)
- **Course name resolution**: When users mention abbreviated course names (like "MCP", "RAG course", etc.), first use the course outline tool to find the full course title, then use the search tool with the complete name
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tools first, then answer
- **Complex queries**: Break down into multiple tool calls as needed
- **Course outline questions**: Use outline tool to return course title, course link, and complete lesson list with numbers and titles
- **Abbreviated course names**: When users use short names like "MCP course", "RAG course", etc., ALWAYS use the course outline tool first to find the full course title, then use content search tool with the complete name
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_sequential_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            return "No response generated"
    
    def _handle_sequential_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle sequential tool execution with up to max_tool_rounds rounds.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Initialize conversation with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0
        
        while (current_response.stop_reason == "tool_use" and 
               round_count < self.max_tool_rounds and 
               tool_manager):
            
            round_count += 1
            
            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute all tool calls in this round
            tool_results = self._execute_tool_round(current_response, tool_manager)
            
            # Handle tool execution failures
            if not tool_results:
                return "Tool execution failed. Please try again."
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
            
            # Make next API call with tools still available
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools", []),
                "tool_choice": {"type": "auto"}
            }
            
            try:
                current_response = self.client.messages.create(**next_params)
            except Exception as e:
                return f"Error in tool execution round {round_count}: {str(e)}"
        
        # If we exited because of max rounds and Claude still wants to use tools,
        # make a final call without tools to get a text response
        if (current_response.stop_reason == "tool_use" and 
            round_count >= self.max_tool_rounds):
            
            # Add a message prompting Claude to synthesize the tool results
            final_messages = messages.copy()
            final_messages.append({
                "role": "user", 
                "content": "Please provide a comprehensive answer based on the search results you found above."
            })
            
            # Make final call without tools to force a text response
            final_params = {
                **self.base_params,
                "messages": final_messages,
                "system": base_params["system"]
                # No tools parameter - forces Claude to provide text response
            }
            
            try:
                current_response = self.client.messages.create(**final_params)
            except Exception as e:
                return f"Error in final response generation: {str(e)}"
        
        # Extract final response text
        return self._extract_response_text(current_response)
    
    def _execute_tool_round(self, response, tool_manager):
        """Execute all tool calls in a single round"""
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution error: {str(e)}",
                        "is_error": True
                    })
        return tool_results
    
    def _extract_response_text(self, response):
        """Safely extract text from API response"""
        if response.content and len(response.content) > 0:
            # Find the first TextBlock in the response content
            for content_block in response.content:
                # Check if it's a text block (has text attribute and is not a tool_use)
                if hasattr(content_block, 'text') and hasattr(content_block, 'type'):
                    if content_block.type != 'tool_use':
                        return content_block.text
                # Fallback: if it has text but no type attribute, assume it's text
                elif hasattr(content_block, 'text'):
                    text_content = content_block.text
                    # Make sure it's actually a string, not another object
                    if isinstance(text_content, str) and text_content.strip():
                        return text_content
            
            # If we reach here, we only found tool_use blocks or empty content
            # This should not happen with our improved max rounds handling
            return "I was able to search for information but encountered an issue generating the final response. Please try rephrasing your question."
        else:
            # Handle the case where response.content is empty or None
            return "I encountered an issue generating a response. Please try rephrasing your question."
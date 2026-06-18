import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information.

Tool Usage:
- **get_course_outline**: Use for outline-related queries — "what lessons does X have?", "show me the outline for X", "what topics are covered in X?". Always respond with the course title, course link, and every lesson number and title.
- **search_course_content**: Use for questions about specific course content or detailed educational materials. You may use up to 2 sequential tool calls when a query requires information from multiple sources (e.g., first retrieving a course outline, then searching for related content based on what you found).
- Synthesize tool results into accurate, fact-based responses.
- If a tool yields no results, state this clearly without offering alternatives.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Outline questions**: Use get_course_outline, then list the course title, course link, and all lessons (number + title)
- **Course-specific content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
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
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tools, tool_manager)

        # Return direct response
        return response.content[0].text

    def _run_tool_loop(
        self, initial_response, base_params: Dict[str, Any], tools, tool_manager
    ) -> str:
        """
        Drive up to MAX_TOOL_ROUNDS sequential tool-call rounds and return final text.

        Each intermediate round keeps tools available so Claude can decide whether
        to make another tool call. The final forced-text call omits tools.
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while current_response.stop_reason == "tool_use" and round_count < self.MAX_TOOL_ROUNDS:
            # Append assistant's tool-use content to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this response
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": result,
                            }
                        )
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution failed: {str(e)}",
                                "is_error": True,
                            }
                        )

            # Append tool results as a single user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Build next API call — include tools for intermediate rounds, omit for the final forced-text round
            is_last_round = round_count == self.MAX_TOOL_ROUNDS - 1
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }
            if not is_last_round and tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**next_params)
            round_count += 1

        return current_response.content[0].text

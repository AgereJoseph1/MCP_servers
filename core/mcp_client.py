import asyncio
import json
from typing import Any
import uuid

from collections import deque
from contextlib import AsyncExitStack
from datetime import UTC, timedelta
from core.logging_config import get_logger


from core.config import settings


from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
from mcp import ClientSession
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessage



class MCPClient:
    def __init__(self, model: str = "gpt-4o", mcp_urls: list[str] = None):
        if mcp_urls is None:
            mcp_urls = [
                #"https://data-modeling.happybay-27a476d7.switzerlandnorth.azurecontainerapps.io/streamable-http/mcp/"
                "https://data-modeling.happybay-27a476d7.switzerlandnorth.azurecontainerapps.io/streamable-http/mcp/"
            ]
        self.model = model
        self.mcp_urls = mcp_urls
        self.sessions: dict[str, ClientSession] = {}
        self._tool_url: dict[str, str] = {}
        self.exit_stack = AsyncExitStack()
        self._initialized = False
        self.model_client = OpenAI(api_key=settings.llm_api_key)
        self.sessions = {}
        self.connected = False
        self.tools = []
        self._tool_url = {}  # <-- Add this line
        self.messages = []
        self.logger = get_logger(__name__)
        self.logger.info(f"[DEBUG] MCPClient initialized with URLs: {self.mcp_urls}")

    async def connect_to_servers(
        self, headers: dict[str, Any] | None = None
    ) -> list[Tool]:
        """Connect to multiple MCP servers with improved error handling"""
        logger = get_logger(__name__ + ".connect_to_servers")
        urls = self.mcp_urls
        tools: list[Tool] = []
        failed_connections = []

        for url in urls:
            try:
                read_stream, write_stream, _ = (
                    await self.exit_stack.enter_async_context(
                        streamablehttp_client(
                            url=url,
                            headers=headers,
                            sse_read_timeout=timedelta(
                                seconds=settings.TOOL_CALL_TIMEOUT + 10
                            ),
                        )
                    )
                )
                session_context = ClientSession(read_stream, write_stream)
                session = await self.exit_stack.enter_async_context(session_context)
                self.sessions[url] = session

                await self.sessions[url].initialize()
                response = await self.sessions[url].list_tools()
                tools.extend(response.tools)

                for tool in response.tools:
                    self._tool_url[tool.name] = url

                logger.info(f"Successfully connected to {url}")

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {url}: {e}")
                failed_connections.append(url)
                # Continue with other servers instead of failing completely

        if failed_connections and not self.sessions:
            raise RuntimeError(
                f"Failed to connect to all MCP servers: {failed_connections}"
            )

        if failed_connections:
            logger.warning(f"Some MCP servers failed to connect: {failed_connections}")

        tools_names = [tool.name for tool in tools]
        logger.info(
            f"Connected to {len(self.sessions)} server(s) with tools: {tools_names}"
        )
        self._initialized = True

        return tools_names

    async def get_tools(self) -> list[Tool]:
        """Gets all tools from the current sessions"""
        if not self._initialized:
            raise RuntimeError(
                "MCP client not initialized. Call connect_to_servers first."
            )

        tools = []
        for url, session in self.sessions.items():
            try:
                response = await session.list_tools()
                tools.extend(response.tools)
            except Exception as e:
                logger = get_logger(__name__ + ".get_tools")
                logger.error(f"Failed to get tools from {url}: {e}")
                # Continue with other sessions

        return tools

    def get_session(self, tool_name: str) -> ClientSession:
        """Get session using tool name"""
        url = self._tool_url.get(tool_name)
        if not url:
            raise ValueError(f"Tool '{tool_name}' not found in available tools")
        session = self.sessions.get(url)
        if not session:
            raise ValueError(f"Session for tool '{tool_name}' not found")

        return session

    async def process_query(
        self,
        messages: list[dict[str, Any]],
        system_prompt: dict[str, str] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Process a query using the LLM and available tools with improved error handling
        """
        logger = get_logger(__name__ + ".process_query")

        if not self._initialized:
            raise RuntimeError(
                "MCP client not initialized. Call connect_to_servers first."
            )

        try:
            # Add system prompt if provided
            if system_prompt:
                
                messages = [system_prompt] + messages

            initial_len = len(messages) - 1

            # Get available tools
            response = await self.get_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in response
            ]

            #print(f'Messages for initial llm call: {messages}')

            # Ensure messages is a flat list of dicts (not a list of lists)
            if any(isinstance(m, list) for m in messages):
                # Flatten one level if needed
                flat_messages = []
                for m in messages:
                    if isinstance(m, list):
                        flat_messages.extend(m)
                    else:
                        flat_messages.append(m)
                messages = flat_messages
            if any(isinstance(m['content'], dict) for m in messages):
                for m in messages:
                    if isinstance(m['content'], dict):
                        m['content'] = json.dumps(m['content'], default=str)

            # Initial LLM call
            response = self.model_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
            )

            final_text = []
            assistant_message = response.choices[0].message
            queue = deque()
            queue.append(assistant_message)

            # Process response queue (handles tool calls and follow-up responses)
            max_iterations = 10  # Prevent infinite loops
            iteration_count = 0

            while len(queue) > 0 and iteration_count < max_iterations:
                iteration_count += 1
                assistant_message: ChatCompletionMessage = queue.popleft()

                if not assistant_message.tool_calls:
                    # Regular text response
                    logger.info(f'assistant msg: {assistant_message}')
                    messages.append(assistant_message.model_dump(exclude_none=True))
                    if assistant_message.content:
                        final_text.append(assistant_message.content)
                else:
                    # Handle tool calls
                    # Add assistant message with tool calls
                    messages.append(assistant_message.model_dump(exclude_none=True))

                    # Execute each tool call
                    for tool_call in assistant_message.tool_calls:
                        try:
                            # Ensure tool call has an ID
                            if not tool_call.id:
                                tool_call.id = str(uuid.uuid4())

                            tool_name = tool_call.function.name

                            # Validate and parse tool arguments
                            if not tool_call.function.arguments:
                                logger.error(f"Tool call {tool_name} has no arguments")
                                raise ValueError(
                                    f"Tool call {tool_name} has no arguments"
                                )

                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError as json_error:
                                logger.error(
                                    f"Invalid JSON in tool arguments for {tool_name}: {json_error}"
                                )
                                # Add error response for this tool call
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"Error: Invalid tool arguments - {str(json_error)}",
                                    }
                                )
                                continue

                            # Get session for tool
                            try:
                                session = self.get_session(tool_name=tool_name)
                            except ValueError as session_error:
                                logger.error(
                                    f"Session error for tool {tool_name}: {session_error}"
                                )
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"Error: {str(session_error)}",
                                    }
                                )
                                continue

                            # Execute tool call
                            logger.info(
                                f"Calling tool {tool_name} with args {tool_args}"
                            )

                            try:
                                result = await session.call_tool(
                                    tool_name,
                                    tool_args,
                                    read_timeout_seconds=timedelta(
                                        seconds=settings.TOOL_CALL_TIMEOUT
                                    ),
                                )

                                logger.info(f'Result from tool call: {result.content}')

                                # Add tool result to messages
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": [
                                            item.model_dump() for item in result.content
                                        ],
                                    }
                                )

                                logger.info(f"Tool {tool_name} executed successfully")

                            except asyncio.TimeoutError:
                                logger.error(f"Tool call {tool_name} timed out")
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": "Error: Tool call timed out",
                                    }
                                )
                            except Exception as tool_error:
                                logger.error(
                                    f"Error executing tool {tool_name}: {tool_error}"
                                )
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"Error: {str(tool_error)}",
                                    }
                                )

                        except Exception as call_error:
                            logger.error(
                                f"Error processing tool call {tool_call.function.name}: {call_error}"
                            )
                            raise RuntimeError(
                                f"Error processing tool call {tool_call.function.name}: {call_error}"
                            )

                            # Continue with other tool calls

                    #Get LLM response after tool execution
                    try:
                        response = self.model_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=available_tools,
                            temperature=0,
                            seed=35
                        )
                        logger.info(f'Response from llm after tool call: {response}')
                        assistant_message = response.choices[0].message
                        queue.append(assistant_message)

                    except Exception as llm_error:
                        logger.error(
                            f"Error getting LLM response after tool calls: {llm_error}"
                        )
                        raise llm_error

            if iteration_count >= max_iterations:
                logger.warning(
                    f"Maximum iterations ({max_iterations}) reached in process_query"
                )
                raise RuntimeError("Response processing reached maximum iterations.")

            # Return final response and new messages
            final_response = "\n".join(filter(None, final_text))
            new_messages = messages[initial_len:]

            return final_response, new_messages

        except OpenAIError as openai_error:
            logger.error(f"OpenAI API error in process_query: {openai_error}")
            raise openai_error
        except Exception as e:
            logger.exception(f"Unexpected error in process_query: {e}")
            raise RuntimeError(f"Query processing failed: {str(e)}") from e

    # async def process_query_stream(
    #     self,
    #     messages: list[dict[str, Any]],
    #     conversation: Conversation,
    #     system_prompt: dict[str, str] | None = None,
    # ) -> AsyncGenerator[str, None]:
    #     """Process a query using the LLM and available tools with improved error handling"""
    #     logger = getLogger(__name__ + ".process_query_stream")

    #     try:
    #         if system_prompt:
    #             messages = [system_prompt] + messages

    #         get_tools_response = await self.get_tools()
    #         available_tools = [
    #             {
    #                 "type": "function",
    #                 "function": {
    #                     "name": tool.name,
    #                     "description": tool.description,
    #                     "parameters": tool.inputSchema,
    #                 },
    #             }
    #             for tool in get_tools_response
    #         ]

    #         stream: Stream[ChatCompletionChunk] = (
    #             self.model_client.chat.completions.create(
    #                 model=self.model,
    #                 messages=messages,
    #                 tools=available_tools,
    #                 stream=True,
    #             )
    #         )

    #         stream_queue = deque()
    #         stream_queue.append(stream)

    #         while len(stream_queue) > 0:
    #             current_stream = stream_queue.popleft()
    #             final_tool_calls = {}
    #             full_assistant_response_content = ""

    #             try:
    #                 # Process stream to get assistant final response or tool calls
    #                 for chunk in current_stream:
    #                     if not chunk.choices:
    #                         continue

    #                     delta = chunk.choices[0].delta

    #                     # Get assistant final response
    #                     if delta.content:
    #                         yield delta.content
    #                         full_assistant_response_content += delta.content

    #                     # Get tool calls
    #                     if delta.tool_calls:
    #                         for tool_call in delta.tool_calls:
    #                             index = tool_call.index
    #                             if index not in final_tool_calls:
    #                                 final_tool_calls[index] = tool_call
    #                             else:
    #                                 # Accumulate function arguments
    #                                 if (
    #                                     tool_call.function
    #                                     and tool_call.function.arguments
    #                                 ):
    #                                     final_tool_calls[
    #                                         index
    #                                     ].function.arguments += (
    #                                         tool_call.function.arguments
    #                                     )

    #             except Exception as stream_error:
    #                 logger.error(f"Error processing stream: {stream_error}")
    #                 yield f"Error processing stream: {str(stream_error)}"
    #                 break

    #             # Handle assistant response
    #             if full_assistant_response_content:
    #                 message = {
    #                     "role": "assistant",
    #                     "content": full_assistant_response_content,
    #                 }
    #                 conversation.messages.append(Message(content=message))
    #                 messages.append(message)

    #             # Handle tool calls
    #             elif final_tool_calls:
    #                 # Prepare tool calls message
    #                 tool_calls_data = []
    #                 for call in final_tool_calls.values():
    #                     tool_calls_data.append(call.model_dump())

    #                 message = {
    #                     "role": "assistant",
    #                     "tool_calls": tool_calls_data,
    #                 }
    #                 messages.append(message)
    #                 conversation.messages.append(Message(content=message))

    #                 # Execute tool calls
    #                 for call in final_tool_calls.values():
    #                     try:
    #                         tool_call: ChoiceDeltaToolCall = call
    #                         tool_name = tool_call.function.name

    #                         # Validate tool call arguments
    #                         if not tool_call.function.arguments:
    #                             logger.error(f"Tool call {tool_name} has no arguments")
    #                             raise ValueError(
    #                                 f"Tool call {tool_name} has no arguments"
    #                             )

    #                         tool_args = json.loads(tool_call.function.arguments)
    #                         session = self.get_session(tool_name=tool_name)

    #                         yield f"Calling tool {tool_name} with args {tool_args}"

    #                         result = await session.call_tool(
    #                             tool_name,
    #                             tool_args,
    #                             read_timeout_seconds=timedelta(
    #                                 seconds=settings.TOOL_CALL_TIMEOUT
    #                             ),
    #                         )

    #                         raw_output = {
    #                             tool_name: [
    #                                 item.model_dump() for item in result.content
    #                             ]
    #                         }
    #                         yield f"{json.dumps(raw_output, default=str)}"

    #                         # Save tool result to conversation
    #                         tool_message = {
    #                             "role": "tool",
    #                             "tool_call_id": tool_call.id,
    #                             "content": [
    #                                 item.model_dump() for item in result.content
    #                             ],
    #                         }
    #                         messages.append(tool_message)
    #                         conversation.messages.append(Message(content=tool_message))

    #                     except json.JSONDecodeError as json_error:
    #                         logger.error(
    #                             f"Invalid JSON in tool arguments for {tool_name}: {json_error}"
    #                         )
    #                         yield f"Error: Invalid tool arguments for {tool_name}"
    #                     except Exception as tool_error:
    #                         logger.error(
    #                             f"Error calling tool {tool_name}: {tool_error}"
    #                         )
    #                         yield f"Error calling tool {tool_name}: {str(tool_error)}"
    #                         # Continue with other tools instead of failing completely

    #                 # Call assistant again to process tool results
    #                 try:
    #                     next_stream: Stream[ChatCompletionChunk] = (
    #                         self.model_client.chat.completions.create(
    #                             model=self.model,
    #                             messages=messages,
    #                             tools=available_tools,
    #                             stream=True,
    #                         )
    #                     )
    #                     stream_queue.append(next_stream)
    #                 except Exception as next_stream_error:
    #                     logger.error(f"Error creating next stream: {next_stream_error}")
    #                     yield f"Error processing tool results: {str(next_stream_error)}"
    #                     break

    #     except Exception as e:
    #         logger.exception(f"Error in process_query_stream: {e}")
    #         yield f"Stream processing error: {str(e)}"

    async def cleanup(self):
        """Clean up resources with error handling"""
        logger = get_logger(__name__ + ".cleanup")
        try:
            if hasattr(self, "exit_stack"):
                await self.exit_stack.aclose()
            logger.info("MCP client cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.__aexit__(exc_type, exc, tb)

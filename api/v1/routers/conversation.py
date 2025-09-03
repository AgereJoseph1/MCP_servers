# """Model chat router for logical data modeling assistant."""

# import asyncio
# import time
# from uuid import UUID, uuid4
# from fastapi import APIRouter, Header, HTTPException, Depends
# from typing import List, Dict, Any, Optional
# from schemas.api import Message, QueryRequest, QueryResponse
# from schemas.data_model import LogicalDataModel
# from core.data_model_service import generate_logical_data
# from core.authentication.auth_middleware import get_current_token
# from schemas.token import TokenData

# from core.config import settings
# from core.storage import (
#     get_chat_history, add_message_to_history,
#     clear_chat_history, get_utc_timestamp,
#     clean_logical_model, chat_histories,
#     order_chat_history
# )
# from core.data_model_service import generate_logical_data, extract_json_from_string
# from core.prompts.main import NEW_MCP_PROMPT
# from core.logging_config import get_logger
# import json
# import uuid
# from fastapi.responses import  StreamingResponse
# from datetime import  UTC
# from core.mcp_client import MCPClient

# # In-memory store for KB references per user (session-aware, not persistent)
# user_kb_references: Dict[str, Dict[str, Any]] = {}

# #mcp_url = "https://data-modeling.happybay-27a476d7.switzerlandnorth.azurecontainerapps.io/streamable-http/mcp/"
# mcp_url = "http://127.0.0.1:8000/streamable-http/mcp/"
# print(f"[DEBUG] Connecting to MCP server at: {mcp_url}")
# mcp_client = MCPClient(
#     mcp_urls= [mcp_url]
# )


# logger = get_logger(__name__)
# router = APIRouter(tags=["Model Chat"])

# # @router.post(
# #     "/create-logical-model",
# #     response_model=LogicalDataModel,
# #     operation_id="create_new_logical_model",
# #     summary="Create a new logical data model (MCP tool endpoint)"
# # )
# # async def create_new_logical_model(request: QueryRequest):
# #     """
# #     MCP Tool Endpoint: Create a new logical data model from a single user request.
# #     This tool generates a logical data model based on the provided query.
# #     Session ID is set internally and not exposed to the client.
# #     """
# #     import uuid
# #     session_id = str(uuid.uuid4())  # Hidden/internal session ID

# #     messages = [
# #         {"role": "system", "content": UPDATED_SYSTEM_PROMPT},
# #         {"role": "user", "content": request.query}
# #     ]
# #     model = generate_logical_data(messages, request.query)
# #     model_dict = model.model_dump() if hasattr(model, 'model_dump') else model
# #     model_clean = clean_logical_model(model_dict)
# #     return model_clean

# # @router.get("/model-chat/history", response_model=QueryResponse, summary="Get the current chat history")
# # async def get_chat_history_endpoint(
# #     user_id: Optional[str] = Header(settings.default_user_id, include_in_schema=False)
# # ) -> QueryResponse:
# #     """
# #     Get chat history for a user.

# #     Args:
# #         user_id: User identifier (from header)

# #     Returns:
# #         QueryResponse with chat history
# #     """
# #     try:
# #         logger.debug(f"Retrieving chat history for user {user_id}")
# #         history = get_chat_history(user_id)
# #         ordered_history = order_chat_history(history)

# #         # Convert history to Message objects with timestamps
# #         response_messages = [Message(**add_timestamp_and_fix_role(msg)) for msg in ordered_history]

# #         logger.debug(f"Retrieved {len(response_messages)} messages for user {user_id}")
# #         return QueryResponse(messages=response_messages)

# #     except Exception as e:
# #         logger.error(f"Error retrieving chat history for user {user_id}: {e}")
# #         raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

# # @router.get("/log/download", summary="Download the application log file")
# # async def download_log_file(
# #     current_user: TokenData = Depends(get_current_token)
# # ):
# #     """
# #     Download the main application log file (app.log).
# #     Returns the log file as an attachment.
# #     """
# #     import os
# #     from io import BytesIO
# #     log_path = os.path.join("logs", "app.log")
# #     try:
# #         if not os.path.exists(log_path) or not os.path.isfile(log_path):
# #             logger.warning(f"Log file not found: {log_path}")
# #             raise HTTPException(status_code=404, detail="Log file not found")
# #         # Read the file into memory to avoid Content-Length issues
# #         with open(log_path, "rb") as f:
# #             file_bytes = f.read()
# #         return StreamingResponse(BytesIO(file_bytes), media_type="text/plain", headers={"Content-Disposition": "attachment; filename=app.log"})
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"Error serving log file: {e}")
# #         raise HTTPException(status_code=500, detail="Failed to download log file")



# # @router.post('/mcpchat', response_model=QueryResponse)
# # async def mcp_model_chat(
#     request: QueryRequest,
#     current_user: TokenData = Depends(get_current_token),
#     # user_id: Optional[str] = Header(settings.default_user_id),
#     session_id: Optional[UUID] = None,
# ) -> QueryResponse:
#     from core.config import settings

#     await mcp_client.connect_to_servers()
#     print("[DEBUG] MCP client connected.")

#     # print("[DEBUG] LLM_BASE_URL:", settings.llm_base_url)
#     # print("[DEBUG] MCP_SERVER_URL:", getattr(settings, 'mcp_server_url', None))
#     if not session_id:
#         session_id = uuid4()
#         print(f"[REFERENCE TRACKING] user_id is missing or default. Reference tracking may not work as expected.")
#     try:
#         system_message = [{"role": "system", "content": NEW_MCP_PROMPT, "timestamp": get_utc_timestamp()}]
#         await add_message_to_history(session_id, {"role": "user", "content": request.query, "timestamp": get_utc_timestamp()})

#         history = await get_chat_history(session_id)
#         logger.info(f' This is the chat history: {history}')

#         logger.info(f' This is the chat history: {history}')
#         logger.info(f' This is the system message: {system_message}')

#         final_response, new_messages = await mcp_client.process_query(
#             messages=history,
#             system_prompt=system_message
#         )
#         print(f"[DEBUG] Final response from MCP client: {final_response}")

#         # Extract only the JSON object from the LLM response
#         json_model = extract_json_from_string(final_response)

#         response = {"role": "assistant", "content": json_model, "timestamp": get_utc_timestamp()}
#         await add_message_to_history(session_id, response)

#         updated_history = await get_chat_history(session_id)
#         ordered_history = order_chat_history(updated_history)

#         for msg in ordered_history:
#             if msg.get("timestamp") is None:
#                 msg["timestamp"] = get_utc_timestamp()

#             try:
#                 msg['content'] = json.loads(msg['content'])
#             except (json.JSONDecodeError, TypeError):
#                 # If content is not JSON, keep it as is
#                 pass

#         print(f'ordered mesg: {ordered_history}')
#         response_messages = [Message(**msg) for msg in ordered_history]
#         return QueryResponse(session_id=session_id,messages=response_messages) 
#     except Exception as e:
#         logger.exception("Unhandled error in /mcpchat endpoint")
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred. Please try again or contact support."
#         )

# @router.post("/mcpchat/reset/{session_id}", summary="Reset the mcpchat session history")
# async def reset_mcpchat(
#      session_id: UUID,
#      current_user: TokenData = Depends(get_current_token),
# ) -> Dict[str, str]:
#     """
#     Reset chat history for a session (used by /mcpchat).
#     """
#     try:
#         if not session_id or not await get_chat_history(session_id):
#             raise HTTPException(status_code=400, detail="No session ID provided")
#         logger.info(f"Resetting mcpchat history for session {session_id}")
#         await clear_chat_history(session_id)
#         return {"message": "Chat history has been reset."}
#     except Exception as e:
#         logger.error(f"Error resetting mcpchat history for session {session_id}: {e}")
#         raise HTTPException(status_code=500, detail="Failed to reset chat history")


# @router.get("/mcpchat/conversations")
# async def get_chat_history_endpoint(
#         session_id: UUID | None  = None,
#         current_user: TokenData = Depends(get_current_token),
# ):
#     """
#     Reset chat history for a session (used by /mcpchat).
#     """
#     try:
#         if session_id:
#             logger.info(f"Resetting mcpchat history for session {session_id}")
#             return await get_chat_history(session_id)
#         logger.info("returning all chat histories")
#         return chat_histories

#     except Exception as e:
#         logger.error(f"Error resetting mcpchat history for session {session_id}: {e}")
#         raise HTTPException(status_code=500, detail="Failed to reset chat history")



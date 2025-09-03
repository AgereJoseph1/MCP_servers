"""Model-related schemas for the logical data modeling assistant."""

from pydantic import BaseModel, Field
from uuid import UUID
from typing import List, Dict, Any, Literal, Optional

class Message(BaseModel):
    """A single message in the chat between user and assistant."""
    role: Literal["user", "assistant", "tool", "system"] = Field(..., description="The role of the message sender: 'user' or 'assistant'.")
    content: Any = Field(..., description="The message content. For 'user', this is a string. For 'assistant', this is the logical data model as a JSON object.")
    timestamp: Optional[str] = Field(..., description="The ISO 8601 UTC timestamp when the message was created.")

class QueryRequest(BaseModel):
    """Request body for the /model-chat endpoint. Only the user's query is required."""
    query: str = Field(..., description="The user's request or instruction for the data modeling assistant.")

class QueryResponse(BaseModel):
    """Response body for the /model-chat endpoint, containing the user query and the assistant's response."""
    session_id: UUID = Field(..., description="The session ID of the user query.")
    messages: List[Message] = Field(..., description="The user query and the assistant's response (logical data model).")

class VectorStoreQueryRequest(BaseModel):
    """Request body for vector store queries."""
    query: str = Field(..., description="The query string to search for in the vector store")
    k: int = Field(default=4, description="Number of results to return")
    pre_filter: Dict[str, Any] = Field(default_factory=dict, description="Pre-filter criteria for the search")

class VectorStoreQueryResponse(BaseModel):
    """Response body for vector store queries."""
    query: str = Field(..., description="The original query string")
    results: List[Any] = Field(..., description="List of search results from the vector store (can be [document, score] tuples or dictionaries)")
    total_results: int = Field(..., description="Total number of results found") 


# --- Update Logical Model Tool Schemas ---

class UpdateInstruction(BaseModel):
    """Instruction describing a single targeted update to the logical model.

    - Identify targets via IDs when possible; names are supported as fallback.
    - Relationships are never modified by this tool.
    """
    action: Literal[
        "add_entity",
        "rename_entity",
        "set_entity_type",
        "set_entity_metadata",
        "add_attribute",
        "update_attribute",
        "remove_attribute",
    ]
    # Target selectors (optional depending on action)
    targetEntityId: Optional[str] = Field(default=None, description="ID of the target entity")
    targetEntityName: Optional[str] = Field(default=None, description="Name of the target entity (fallback if ID not provided)")
    targetAttributeId: Optional[str] = Field(default=None, description="ID of the target attribute (for attribute actions)")
    targetAttributeName: Optional[str] = Field(default=None, description="Name of the target attribute (fallback if ID not provided)")
    # Free-form payload for the action (e.g., new names, attribute definitions)
    payload: Dict[str, Any] | None = Field(default=None, description="Action-specific data for the update")


class UpdateLogicalModelRequest(BaseModel):
    """Request to update parts of an existing `LogicalPhysicalModel` while preserving relationships."""
    current_model: Dict[str, Any] = Field(..., description="The current LogicalPhysicalModel JSON to update")
    updates: List[UpdateInstruction] = Field(..., description="List of targeted updates to apply")
    preserve_relationships: bool = Field(default=True, description="Must remain True; tool will error if an update would break relationships")


class UpdateLogicalModelLLMRequest(BaseModel):
    """Request to update the logical model via LLM with a natural language instruction."""
    current_model: Dict[str, Any] = Field(..., description="The current LogicalPhysicalModel JSON to update")
    instruction: str = Field(..., description="Natural language instruction describing exactly what to update")


# --- Session-based Update Requests (pull model from conversation history) ---

class SessionUpdateLogicalModelRequest(BaseModel):
    """Update the latest model in a session using targeted structured updates.

    The server will fetch the latest assistant-produced model from the conversation
    history for the given `session_id`, then apply the provided updates while
    preserving relationships.
    """
    session_id: UUID = Field(..., description="Conversation session ID containing the model to update")
    updates: List[UpdateInstruction] = Field(..., description="List of targeted updates to apply")
    preserve_relationships: bool = Field(default=True, description="Must remain True; updates won't modify relationships")


class SessionUpdateLogicalModelLLMRequest(BaseModel):
    """Update the latest model in a session using a natural language instruction via LLM.

    The server will fetch the latest assistant-produced model from the conversation
    history for the given `session_id`, then apply the natural language instruction.
    """
    session_id: str = Field(..., description="Conversation session ID containing the model to update")
    instruction: str = Field(..., description="Natural language instruction for the update")
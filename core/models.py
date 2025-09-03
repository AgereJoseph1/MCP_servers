import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Relationship, SQLModel
from uuid_extensions import uuid7


# region Agent Workforce Link
class AgentWorkforceLink(SQLModel, table=True):
    agent_id: uuid.UUID | None = Field(
        default=None, foreign_key="agent.id", primary_key=True
    )
    workforce_id: uuid.UUID | None = Field(
        default=None, foreign_key="workforce.id", primary_key=True
    )


# endregion


# region Workforce


class WorkforceBase(SQLModel):
    name: str
    model: str = Field(default="gpt-4o")
    description: str
    instruction: str
    workflow_type: str = Field(default="coordinator")
    # mcp_server: str | None = Field(default=None)
    max_iterations: int | None = Field(default=None)


class Workforce(WorkforceBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    agents: list["Agent"] = Relationship(
        back_populates="workforces", link_model=AgentWorkforceLink
    )


class WorkforceCreate(WorkforceBase):
    sub_agent_ids: list[uuid.UUID]


class WorkforcePublic(WorkforceBase):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agents: list["AgentPublic"]


class WorkforceUpdate(SQLModel):
    name: str | None = None
    model: str | None = None
    description: str | None = None
    instruction: str | None = None
    workflow_type: str | None = None
    # mcp_server: str | None = None
    max_iterations: int | None = None

    sub_agent_ids: list[uuid.UUID] | None = None


# endregion


# region Agent
class AgentBase(SQLModel):
    name: str
    model: str = Field(default="gpt-4o")
    description: str
    instruction: str
    # mcp_server: str | None = Field(default=None)


class Agent(AgentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    workforces: list["Workforce"] = Relationship(
        back_populates="agents", link_model=AgentWorkforceLink
    )
    mcp_servers: list["MCPServer"] = Relationship(
        back_populates="agent", cascade_delete=True
    )
    conversations: list["Conversation"] = Relationship(
        back_populates="agent",
        cascade_delete=True,
    )


class AgentCreate(AgentBase):
    mcp_servers: list["MCPServerBase"]


class AgentPublic(AgentBase):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str | None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    mcp_servers: list["MCPServerPublic"]
    # conversations: list["ConversationPublic"]


class AgentUpdate(SQLModel):
    name: str | None = None
    model: str | None = None
    description: str | None = None
    instruction: str | None = None
    mcp_servers: list["MCPServerBase"] | None = None


# endregion


# region MCP Server
class MCPServerBase(SQLModel):
    name: str | None = Field(default=None)
    url: str


class MCPServer(MCPServerBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)

    agent_id: uuid.UUID | None = Field(
        default=None, foreign_key="agent.id", ondelete="CASCADE"
    )
    agent: Agent | None = Relationship(back_populates="mcp_servers")

    mcp_server_tools: list["MCPServerTool"] = Relationship(
        back_populates="mcp_server", cascade_delete=True
    )


class MCPServerPublic(MCPServerBase):
    mcp_server_tools: list["MCPServerToolPublic"]


# endregion


# region MCPServerTool
class MCPServerToolBase(SQLModel):
    name: str


class MCPServerTool(MCPServerToolBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)

    mcp_server_id: uuid.UUID | None = Field(
        default=None, foreign_key="mcpserver.id", ondelete="CASCADE"
    )
    mcp_server: MCPServer | None = Relationship(back_populates="mcp_server_tools")


class MCPServerToolPublic(MCPServerToolBase):
    pass


# endregion


# region Message


class MessageBase(SQLModel):
    content: dict[str, Any] | None = Field(sa_column=Column(JSONB))
    # content: JSONB


class Message(MessageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    conversation_id: uuid.UUID = Field(
        foreign_key="conversation.id", ondelete="CASCADE"
    )

    conversation: Optional["Conversation"] = Relationship(back_populates="messages")


class MessagePublic(MessageBase):
    pass


# endregion


# region Conversation


class ConversationBase(SQLModel):
    title: str | None = Field(default=None)


class Conversation(ConversationBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str
    agent_id: uuid.UUID | None = Field(
        foreign_key="agent.id", default=None, ondelete="CASCADE"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    messages: list[Message] = Relationship(
        back_populates="conversation",
        cascade_delete=True,
    )

    agent: Agent | None = Relationship(back_populates="conversations")


class ConversationPublic(ConversationBase):
    id: uuid.UUID = Field(default_factory=uuid7, primary_key=True)
    user_id: str
    agent_id: uuid.UUID | None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    messages: list[MessagePublic]
    agent: AgentPublic | None


# endregion

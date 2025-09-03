"""Logical data model schemas."""


from pydantic import BaseModel, Field
from typing import List, Optional, Any

class Attribute(BaseModel):
    id: str
    name: str
    type: str
    isPrimaryKey: bool = False
    isForeignKey: bool = False

class Position(BaseModel):
    x: int
    y: int

class Entity(BaseModel):
    id: str 
    name: str
    attributes: List[Attribute]

class Relationship(BaseModel):
    id: str
    fromEntity: str
    toEntity: str
    type: str
    name: str

class LogicalDataModel(BaseModel):
    id: str
    name: str
    message: str
    entities: List[Entity]
    relationships: List[Relationship] 

class UseCase(BaseModel):
    name: str
    definition: str
    description: str

class LogPhysEntity(BaseModel):
    id: str 
    name: str
    type: Optional[str] = Field(None, description="Describes whether this entity is from a logical model or from a physical model")
    attributes: List[Attribute]
    # New fields for physical table metadata
    tableName: Optional[str] = None
    systemName: Optional[str] = None
    environmentName: Optional[str] = None

class LogicalPhysicalModel(BaseModel):
    id: str 
    name: str 
    message: str = Field(..., description="A conversational response explaining its output")
    entities: List[LogPhysEntity]
    relationships: List[Relationship]
    useCase: UseCase 

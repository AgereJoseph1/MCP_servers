from pydantic import BaseModel

from schemas.data_model import Entity, LogPhysEntity

class Columns(BaseModel):
    name: str
    datatype: str | None = None
    definition: str | None = None
    comments: str | None = None
    match: float | None = None

class Metadata(BaseModel):
    systemId: int
    systemName: str
    environmentId: str
    environmentName: str
    authorize: int
    environmentMatch: float | None = None
    tableName: str
    tableComments: str | None = None
    columnCount: int
    columns: list[Columns]

class LMSQueryDocument(BaseModel):
    """Request body for the LMS service query endpoint."""
    id: str
    entity: LogPhysEntity | None = None
    metadata: Metadata

class LMSQueryRequest(BaseModel):
    model_name: str | None = None
    entity: Entity
    limit: int = 25
    w_base: float = 0.25
    w_columns: float = 0.4
    w_auth: float = 0.3 



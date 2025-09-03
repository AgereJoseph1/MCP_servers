from pydantic import BaseModel

class BusinessObject(BaseModel):
    endPointType: str = "SOURCE"
    nodeType: str = "BGM_CUSTOM"
    catalogId: int = 1051
    name: str
    definition: str
    description: str
    objectTypeId: int = 125

class BusinessAssetCreate(BaseModel):
    businessObjects: list[BusinessObject]

class Relationship(BaseModel):
    titleForward: str = "(empty)"
    description: str

class Association(BaseModel):
    sourceObjectId: int
    sourceObjectTypeId: int
    targetObjectId: int
    targetObjectTypeId: int
    relationship: Relationship


class AssociationCreate(BaseModel):
    associations: list[Association]
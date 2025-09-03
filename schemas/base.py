from datetime import timedelta
from typing import Annotated

from bson.errors import InvalidId
from bson.objectid import ObjectId
from core.authentication.auth_token import create_access_token
from pydantic import AfterValidator, BeforeValidator

PyObjectID = Annotated[str, BeforeValidator(str)]

FileUrlID = str


def is_valid_obj_id(value: str) -> str:
    try:
        ObjectId(value)
    except Exception as ex:
        raise ValueError(f"{value} is not a valid object ID")

    return value


ValidIdStr = Annotated[str, AfterValidator(is_valid_obj_id)]


def get_download_link(id: str) -> str:
    """Gets the download link for a file id"""
    payload = {
        "sub": id,
        "id": id,
        "type": "file_access",
        "role": "none",
    }
    token = create_access_token(data=payload, expires_delta=timedelta(days=2))
    res = f"/api/v1/files/{token}/download"

    return res

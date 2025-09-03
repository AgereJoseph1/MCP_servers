from fastapi import APIRouter,HTTPException 
from pydantic import BaseModel
from typing import List,Dict,Any
import httpx

router  = APIRouter(tags=["Data Assets Management"])

# The Erwin upstream API
REMOTE_URL = "http://51.103.210.156:8080/ErwinAIService/api/beta/v1/automateAssetCreation"


# Accept any JSON payload with "content"
class AssetCreationRequest(BaseModel):
    content: Dict[str, Any]


@router.post("/automateAssetCreation")
async def automate_asset_creation(body: AssetCreationRequest):
    """
    Accepts the JSON schema and forwards it to the Erwin API.
    """

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                REMOTE_URL,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=body.model_dump(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream connection error: {exc}")

    # Return response transparently
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return {"status_code": resp.status_code, "text": resp.text}
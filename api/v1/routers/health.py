"""Health check router."""

from fastapi import APIRouter
from schemas.health import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    return HealthResponse(status="healthy", message="Service is running") 
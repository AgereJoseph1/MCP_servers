"""Main FastAPI application for the Logical Data Modeling Assistant."""

import asyncio
import time
import httpx

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from datetime import datetime, UTC
from fastapi import FastAPI, Request,responses
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastmcp import FastMCP
try:
    from fastapi_mcp import FastApiMCP
    _FASTAPI_MCP_AVAILABLE = True
except Exception:  # ImportError or any runtime import issues
    _FASTAPI_MCP_AVAILABLE = False
from fastmcp.server.openapi import MCPType, RouteMap
import uvicorn

from api.v1.routers import conversation, health, lms, data_assets
from api.v1.routers.mcp_tools import router as mcp_tools_router
from core.config import settings
from core.logging_config import setup_logging, get_logger
from core.exceptions import DataModelingException, LLMServiceException, ValidationException, StorageException
from core.lms_service import close_http_client

# from api.v1.routers import conversation, health, lms
from api.v1.routers.mcp_tools import router as mcp_tools_router


# Setup logging
setup_logging()
logger = get_logger(__name__)

# Only include mcp_tools_router in the MCP app for MCP tool exposure
mcp_app = FastAPI()
mcp_app.include_router(mcp_tools_router, prefix="/mcp")

# Wrap with FastMCP
# Include routes with "MCP Tools" tag by excluding everything else
route_maps = [RouteMap(tags={"not_mcp"}, mcp_type=MCPType.EXCLUDE)]
mcp = FastMCP.from_fastapi(app=mcp_app, route_maps=route_maps)
mcp_http = mcp.http_app(path="/mcp")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan manager for startup and shutdown."""
#     # Startup
#     logger.info(" Starting Autonomous Data Product Creation API")
#     logger.info(f" Environment: {'Development' if settings.debug else 'Production'}")
#     logger.info(f" Debug mode: {settings.debug}")
#     logger.info(f" Host: {settings.host}:{settings.port}")
    
#     yield
    
#     # Shutdown
#     logger.info("Shutting down Autonomous Data Product Creation API")
#     await close_http_client()
#     logger.info("Cleanup completed")

# Create FastAPI app with lifespan management
app = FastAPI(
    title=settings.app_name,
    description="Generate and iteratively refine logical data models via chat.",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=mcp_http.lifespan  # Use MCP HTTP app's lifespan for FastMCP compatibility
)

# Add security middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # Configure appropriately for production
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

#usage report middleware for tracking usage
async def save_usage_report(request: Request) -> bool:
    logger.info("save_usage_report")
    try:

        if request.headers.get("Authorization") is None:
            return True

        USAGE_REPORT_ENDPOINT = settings.usage_report_endpoint

        data = {
            "service": "quest_software",  # The service name
            "method": request.method,
            "endpoint": str(request.url),
            "timestamp": datetime.now(UTC).timestamp(),  # Optional
            "ip_address": request.client.host,  # Optional
        }

        headers = {"Authorization": request.headers.get("Authorization")}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                USAGE_REPORT_ENDPOINT, json=data, headers=headers
            )
            return response.is_success
    except Exception as ex:
        logger.error(ex)
        return False


@app.middleware("http")
async def track_usage_middleware(request: Request, call_next):
    task = asyncio.create_task(save_usage_report(request))
    response = await call_next(request)

    return response

 



# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Global exception handler
@app.exception_handler(DataModelingException)
async def data_modeling_exception_handler(request: Request, exc: DataModelingException):
    """Handle custom application exceptions."""
    logger.error(f"Data modeling exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.exception_handler(LLMServiceException)
async def llm_service_exception_handler(request: Request, exc: LLMServiceException):
    """Handle LLM service exceptions."""
    logger.error(f"LLM service exception: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "Service temporarily unavailable", "detail": str(exc)}
    )

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation exceptions."""
    logger.warning(f"Validation exception: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Bad request", "detail": str(exc)}
    )

@app.exception_handler(StorageException)
async def storage_exception_handler(request: Request, exc: StorageException):
    """Handle storage exceptions."""
    logger.error(f"Storage exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

@app.get("/", include_in_schema=False)
async def root():
    return responses.RedirectResponse(url="/docs")

# Include routers


# Expose MCP tools as standard HTTP endpoints as well
app.include_router(mcp_tools_router, prefix="/api/v1/mcp-tools")

# Mount the MCP HTTP app in the main app (only once)
app.mount("/streamable-http", mcp_http)

# Optionally mount FastApiMCP side-by-side for A/B testing without impacting existing endpoints
if _FASTAPI_MCP_AVAILABLE:
    try:
        fastapi_mcp = FastApiMCP(
            app,
            include_operations=[
                "reference_fibo",
                "create_logical_model",
                "add_generated_data_product_to_erwin",
                "update_logical_model",
            ],
        )
        # Mount under a different base path to avoid conflicts with existing FastMCP mount
        fastapi_mcp.mount_http(mount_path="/streamable-http2/mcp")
        logger.info("FastApiMCP mounted at /streamable-http2/mcp")
    except Exception as e:
        logger.warning(f"FastApiMCP initialization failed: {e}")
else:
    logger.info("FastApiMCP not installed; skipping optional MCP mount.")

if __name__ == "__main__":
  
    logger.info("Starting application server...")
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        log_level=settings.log_level.lower()
    ) 

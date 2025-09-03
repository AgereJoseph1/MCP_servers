"""Configuration settings for the application."""

import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings with validation."""
    
    def __init__(self):
        # Application settings
        self.app_name: str = "Logical Data Modeling Assistant API"
        self.app_version: str = "1.0.0"
        self.debug: bool = self._get_bool_env("DEBUG", True)  # Default to True for development
        
        # Server settings
        self.lms_logical_models_store: str = "https://language-model-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/vector-store/lms_store/query"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.mcp_server_url: str = os.getenv("MCP_SERVER_URL", 
                                             "https://data-modeling.happybay-27a476d7.switzerlandnorth.azurecontainerapps.io/streamable-http/mcp/")
        
        # Default user ID for demo purposes 
        self.default_user_id: str = os.getenv("DEFAULT_USER_ID", "demo-user")
        
        # LLM API Configuration
        self.llm_base_url: str = os.getenv(
            "LLM_BASE_URL",
            "https://api.openai.com/v1/"
        )
        self.llm_api_key: str = os.getenv("OPENAI_API_KEY", "")
        # Log the key (masked) at startup for debugging
        if not self.llm_api_key:
            print("[ERROR] OPENAI_API_KEY is not set! LLM calls will fail.")
        else:
            print(f"[INFO] OPENAI_API_KEY loaded: {self.llm_api_key[:6]}...{self.llm_api_key[-4:]}")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
        self.llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "9000"))
        
        # CORS Configuration
        self.lms_vector_store: str = os.getenv(
            "LMS_VECTOR_STORE",
            "https://language-model-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/vector-store/QuestDataAuthorize/query"
        )

        self.lms_embedding_url: str = os.getenv(
            "LMS_EMBEDDING_URL",
            "https://language-model-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/embed/texts"
        )

        # FIBO Vector Store Configuration
        self.fibo_vector_store_url: str = os.getenv(
            "FIBO_VECTOR_STORE_URL",
            "https://language-model-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/vector-store/fibo_rag/query"
        )

        cors_origins_str = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins: List[str] = [cors_origins_str] if cors_origins_str != "*" else ["*"]
        self.cors_credentials: bool = self._get_bool_env("CORS_CREDENTIALS", True)
        self.cors_methods: List[str] = ["*"]  # Simplified for now
        self.cors_headers: List[str] = ["*"]  # Simplified for now
        
        # Logging Configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file_size: int = int(os.getenv("LOG_FILE_SIZE", str(10*1024*1024)))  # 10MB
        self.log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
        
        # Security settings
        self.max_request_size: int = int(os.getenv("MAX_REQUEST_SIZE", str(1024*1024)))  # 1MB
        self.rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        self.TOOL_CALL_TIMEOUT: int = 100000

        # JWT Configuration for Authentication
       
        self.QUEST_SOFTWARE_KEY: str = os.getenv("QUEST_SOFTWARE_KEY", "632364424373493869f96d0257ea08dbc9c95472106614290f145b38d6d68376")
        self.ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_DAYS: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "30"))

        self.mcp_server_url: str = os.getenv(
            "MCP_SERVER_URL",
            "https://data-modeling.happybay-27a476d7.switzerlandnorth.azurecontainerapps.io/streamable-http/mcp/"
        )

        self.mongo_url: str = os.getenv(
            "MONGO_URL",
            "mongodb://74.242.137.125:27018/?directConnection=true"
        )

        self.usage_report_endpoint: str = os.getenv(
            "USAGE_REPORT_ENDPOINT",
            "https://accounts-micro-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/usage_reports"
        )
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

# Create settings instance
settings = Settings()

# Backward compatibility
LMS_VECTOR_STORE = settings.lms_vector_store
USAGE_REPORT_ENDPOINT = settings.usage_report_endpoint
LMS_LOGICAL_MODELS_STORE = settings.lms_logical_models_store
FIBO_VECTOR_STORE_URL = settings.fibo_vector_store_url
DEFAULT_USER_ID = settings.default_user_id
LLM_BASE_URL = settings.llm_base_url
LLM_API_KEY = settings.llm_api_key
LLM_MODEL = settings.llm_model
LLM_MAX_TOKENS = settings.llm_max_tokens
CORS_ORIGINS = settings.cors_origins
CORS_CREDENTIALS = settings.cors_credentials
CORS_METHODS = settings.cors_methods
CORS_HEADERS = settings.cors_headers

# JWT Configuration

QUEST_SOFTWARE_KEY = settings.QUEST_SOFTWARE_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_DAYS = settings.ACCESS_TOKEN_EXPIRE_DAYS 
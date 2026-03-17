import logging
import os
from typing import Literal
from dotenv import load_dotenv
load_dotenv()  # Load .env file before reading os.environ

class Settings:
    APP_NAME: str = "CBSA Backend"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    WEBSOCKET_ENDPOINT: str = "/ws/behaviour"

    # Layer 3 GAT settings (cloud endpoint no longer used; GAT is now in-process)
    DEBUG_MODE: bool = False  # Set to True for local development
    GAT_CLOUD_ENDPOINT: str = ""  # Kept for backwards compatibility; unused
    GAT_WINDOW_SIZE: int = 32  # Deprecated: event count window (kept for compatibility)
    GAT_WINDOW_SECONDS: int = 20  # Temporal graph window in seconds
    GAT_NODE_FEATURE_DIM: int = 56  # 48 behavioral + 8 event-type embedding (device info removed)
    GAT_EDGE_DISTINCT_TARGET: int = 4  # Distinct event types to reach per node
    GAT_ESCALATION_THRESHOLD: float = 0.5  # Assume Layer 2 escalates at this threshold
    GAT_INFERENCE_INTERVAL_SECONDS: float = 5.0  # Interval between GAT inference calls

    # Azure Cosmos DB – shared connection settings for all containers.
    # Used by: cosmos_logger (computation-logs), cosmos_profile_store (user-profiles),
    #          enrollment_store (enrollment-state).
    # Values are read from environment variables at runtime so secrets are
    # never hard-coded in source.
    COSMOS_ENDPOINT: str = os.environ.get("COSMOS_ENDPOINT", "")
    COSMOS_KEY: str = os.environ.get("COSMOS_KEY", "")
    COSMOS_DATABASE: str = os.environ.get("COSMOS_DATABASE", "cbsa-logs")
    COSMOS_CONTAINER: str = os.environ.get("COSMOS_CONTAINER", "computation-logs")
    COSMOS_PROFILES_CONTAINER: str = os.environ.get("COSMOS_PROFILES_CONTAINER", "user-profiles")
    COSMOS_ENROLLMENT_CONTAINER: str = os.environ.get("COSMOS_ENROLLMENT_CONTAINER", "enrollment-state")
    COSMOS_PROTOTYPE_CONTAINER: str = os.environ.get("COSMOS_PROTOTYPE_CONTAINER", "prototype-store")
    COSMOS_BEHAVIOUR_LOGS_CONTAINER: str = os.environ.get("COSMOS_BEHAVIOUR_LOGS_CONTAINER", "behaviour-logs")
    COSMOS_USERS_CONTAINER: str = os.environ.get("COSMOS_USERS_CONTAINER", "users")
    COSMOS_QUARANTINE_CONTAINER: str = os.environ.get("COSMOS_QUARANTINE_CONTAINER", "quarantine-pool")

    # Default adaptive sigma: 0.15 * sqrt(48) — matches drift_engine._DEFAULT_SIGMA
    DEFAULT_ADAPTIVE_SIGMA: float = float(os.environ.get("DEFAULT_ADAPTIVE_SIGMA", "1.0392304845413265"))

    # Admin authorization token – required to call destructive or training endpoints.
    # Set ADMIN_TOKEN in the environment; if empty, admin endpoints are disabled.
    ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "")

    # Azure Blob Storage – model checkpoint files (.pth).
    # Used by: blob_model_store, gat_engine (download on startup).
    AZURE_STORAGE_CONNECTION_STRING: str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    AZURE_STORAGE_CONTAINER: str = os.environ.get("AZURE_STORAGE_CONTAINER", "cbsa-models")


settings = Settings()


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

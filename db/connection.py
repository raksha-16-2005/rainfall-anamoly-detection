"""MongoDB Atlas connection module for rainfall anomaly detection backend."""

import logging
import os
from typing import Optional

import pymongo
from dotenv import load_dotenv
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Module-level singleton client
_client: Optional[pymongo.MongoClient] = None
_db: Optional[pymongo.database.Database] = None
_initialized_collections: set = set()


def _get_client() -> pymongo.MongoClient:
    """
    Get or create the MongoDB client singleton.

    Returns:
        pymongo.MongoClient: Connected MongoDB client instance.

    Raises:
        RuntimeError: If MONGO_URI environment variable is not set.
    """
    global _client

    if _client is not None:
        return _client

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        error_msg = "MONGO_URI environment variable is not set"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        _client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Verify connection
        _client.admin.command("ping")
        logger.info("Successfully connected to MongoDB Atlas")
        return _client
    except ServerSelectionTimeoutError as e:
        logger.error(f"Failed to connect to MongoDB Atlas: {e}")
        raise RuntimeError(f"MongoDB connection failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error connecting to MongoDB: {e}")
        raise RuntimeError(f"MongoDB connection error: {e}") from e


def _get_database() -> pymongo.database.Database:
    """
    Get or create the MongoDB database instance.

    Returns:
        pymongo.database.Database: MongoDB database instance.
    """
    global _db

    if _db is not None:
        return _db

    client = _get_client()
    database_name = os.getenv("MONGO_DB_NAME", "rainfall_detection")
    _db = client[database_name]
    logger.info(f"Using database: {database_name}")
    return _db


def _initialize_collections() -> None:
    """
    Initialize collections and create indexes on first call.

    This function creates the required collections and sets up indexes
    for predictions, alerts, and districts collections.
    """
    global _initialized_collections

    db = _get_database()

    # Predictions collection
    if "predictions" not in _initialized_collections:
        try:
            predictions = db["predictions"]
            # Create compound unique index on (district, date)
            predictions.create_index(
                [("district", pymongo.ASCENDING), ("date", pymongo.ASCENDING)],
                unique=True,
            )
            _initialized_collections.add("predictions")
            logger.info("Initialized 'predictions' collection with indexes")
        except Exception as e:
            logger.error(f"Error initializing 'predictions' collection: {e}")
            raise

    # Alerts collection
    if "alerts" not in _initialized_collections:
        try:
            alerts = db["alerts"]
            # Create index on (risk_level, date)
            alerts.create_index(
                [("risk_level", pymongo.ASCENDING), ("date", pymongo.DESCENDING)]
            )
            _initialized_collections.add("alerts")
            logger.info("Initialized 'alerts' collection with indexes")
        except Exception as e:
            logger.error(f"Error initializing 'alerts' collection: {e}")
            raise

    # Districts collection
    if "districts" not in _initialized_collections:
        try:
            db["districts"]
            _initialized_collections.add("districts")
            logger.info("Initialized 'districts' collection")
        except Exception as e:
            logger.error(f"Error initializing 'districts' collection: {e}")
            raise


def get_collection(name: str) -> Collection:
    """
    Get a MongoDB collection by name.

    On first call, this function initializes all required collections
    and creates necessary indexes.

    Args:
        name (str): The name of the collection to retrieve.

    Returns:
        pymongo.collection.Collection: The MongoDB collection instance.

    Raises:
        RuntimeError: If MongoDB connection fails or MONGO_URI is not set.
    """
    if not _initialized_collections:
        _initialize_collections()

    db = _get_database()
    return db[name]


def ping_db() -> bool:
    """
    Check MongoDB connection health.

    Returns:
        bool: True if connection is healthy, False otherwise.
    """
    try:
        client = _get_client()
        client.admin.command("ping")
        logger.debug("Database health check passed")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def close_connection() -> None:
    """
    Close the MongoDB connection.

    This should be called during application shutdown to properly
    clean up resources.
    """
    global _client, _db, _initialized_collections

    if _client is not None:
        try:
            _client.close()
            _client = None
            _db = None
            _initialized_collections.clear()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

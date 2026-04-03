import logging
from typing import Any

from .env_settings import EnvSettings

has_numpy: bool = False
try:
    import numpy as np

    has_numpy = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def similarity_score(embedding1: dict[str, Any], embedding2: dict[str, Any]) -> float:
    """
    Calculate the cosine similarity between two embeddings.
    Each embedding dict should have the form:
    {
        "embedding": [float, float, ...],
        "text": "some text",
        "dimensions": int
    }
    Returns a value in [-1.0, 1.0], where 1.0 means identical direction.
    """
    if not has_numpy:
        logger.error(
            "NumPy is required for similarity calculations but is not installed. Install with poetry add 'ai-api-unified[similarity_score]'"
        )
        raise ImportError("NumPy is required for similarity calculations.")

    # Extract the raw vector lists; default to empty list if key is missing
    vector1: list[float] = embedding1.get("embedding", [])
    vector2: list[float] = embedding2.get("embedding", [])

    # If either embedding is missing or empty, log an error and return 0.0
    if not vector1 or not vector2:
        logger.error("Embeddings missing for similarity comparison.")
        return 0.0

    # If the two vectors differ in length, they can't be compared; log and exit
    if len(vector1) != len(vector2):
        logger.error(
            "Embedding dimension mismatch: %s vs %s", len(vector1), len(vector2)
        )
        return 0.0

    # Convert Python lists to NumPy arrays of floats for efficient math
    arr1 = np.array(vector1, dtype=float)
    arr2 = np.array(vector2, dtype=float)

    # Compute the Euclidean norm (magnitude) of each vector
    norm1: float = float(np.linalg.norm(arr1))
    norm2: float = float(np.linalg.norm(arr2))

    # Guard against zero-length vectors to avoid division by zero
    if norm1 == 0.0 or norm2 == 0.0:
        logger.error("Zero-length embedding encountered during comparison.")
        return 0.0

    # Compute cosine similarity: (arr1 · arr2) / (‖arr1‖ * ‖arr2‖)
    similarity: float = float(np.dot(arr1, arr2) / (norm1 * norm2))

    return similarity


def is_hex_enabled() -> bool:
    """Check if the we are running in Hex or not based on an environment variable."""

    value: Any = EnvSettings().get_setting("IS_HEX_ENABLED", "false")
    return str(value).lower() in (
        "1",
        "true",
        "yes",
    )

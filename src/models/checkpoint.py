"""Checkpoint data model for rollback functionality."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Checkpoint:
    """Represents a system checkpoint for rollback capability."""

    timestamp: float
    data: Dict[str, Any]
    transaction_id: str
    description: str

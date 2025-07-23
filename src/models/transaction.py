"""Transaction-related data models and enums."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
from .checkpoint import Checkpoint


class TransactionState(Enum):
    """Enumeration of possible transaction states."""

    ACTIVE = "active"
    PREPARING = "preparing"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class Transaction:
    """Represents a database transaction with ACID properties."""

    id: str
    state: TransactionState
    operations: List[Dict[str, Any]]
    timestamp: float
    checkpoint_before: Optional[Checkpoint] = None

"""Checkpoint management for rollback functionality."""

import time
import copy
import threading
from typing import Dict, Any, List
from .models.checkpoint import Checkpoint


class CheckpointManager:
    """Manages system checkpoints for rollback capability."""

    def __init__(self, max_checkpoints: int = 10):
        self.checkpoints: List[Checkpoint] = []
        self.max_checkpoints = max_checkpoints
        self.checkpoint_counter = 0
        self.lock = threading.RLock()

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint transaction ID."""
        self.checkpoint_counter += 1
        return f"checkpoint_{self.checkpoint_counter}_{int(time.time() * 1000)}"

    def create_checkpoint(
        self, data: Dict[str, Any], description: str = ""
    ) -> Checkpoint:
        """Create a checkpoint for rollback capability."""
        with self.lock:
            checkpoint = Checkpoint(
                timestamp=time.time(),
                data=copy.deepcopy(data),
                transaction_id=self._generate_checkpoint_id(),
                description=description,
            )

            self.checkpoints.append(checkpoint)

            # Maintain maximum number of checkpoints
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints.pop(0)

            print(f"Checkpoint created: {description} at {checkpoint.timestamp}")
            return checkpoint

    def get_checkpoint(self, index: int = -1) -> Checkpoint:
        """Get checkpoint by index (default: latest)."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        if index < 0:
            index = len(self.checkpoints) + index

        if index < 0 or index >= len(self.checkpoints):
            raise ValueError(f"Invalid checkpoint index: {index}")

        return self.checkpoints[index]

    def get_checkpoint_count(self) -> int:
        """Get total number of checkpoints."""
        return len(self.checkpoints)

    def get_latest_checkpoint_description(self) -> str:
        """Get description of the latest checkpoint."""
        if not self.checkpoints:
            return "No checkpoints"
        return self.checkpoints[-1].description

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.checkpoints) > 0

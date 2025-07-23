"""Main database class implementing ACID transactions and rollback."""

import copy
import time
import threading
from typing import Dict, Any, Optional
from .transaction_manager import TransactionManager
from .checkpoint_manager import CheckpointManager
from .inventory_operations import InventoryOperations


class AcidRollbackDB:
    """
    Database implementation with Combined Availability Tactics:
    - ACID Transactions for fault prevention
    - Rollback mechanism for fault recovery
    """

    def __init__(self, checkpoint_interval: int = 5, max_checkpoints: int = 10):
        # Core data storage
        self.data: Dict[str, Any] = {}
        self.lock = threading.RLock()

        # Component managers
        self.transaction_manager = TransactionManager()
        self.checkpoint_manager = CheckpointManager(max_checkpoints)

        # Configuration
        self.checkpoint_interval = checkpoint_interval

        # Create initial checkpoint
        self.checkpoint_manager.create_checkpoint(self.data, "initial_state")

        self.inventory = InventoryOperations(self)

    def begin_transaction(self) -> str:
        """Begin a new ACID transaction."""
        with self.lock:
            # Create checkpoint before transaction starts
            checkpoint = self.checkpoint_manager.create_checkpoint(
                self.data, f"pre_transaction"
            )

            return self.transaction_manager.begin_transaction(checkpoint)

    def put(self, txn_id: str, key: str, value: Any) -> bool:
        """Put operation within a transaction."""
        with self.lock:
            if not self.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} not found or not active")

            # Record operation for potential rollback
            operation = {
                "type": "put",
                "key": key,
                "new_value": value,
                "old_value": self.data.get(key),
                "timestamp": time.time(),
            }

            self.transaction_manager.add_operation(txn_id, operation)

            # Apply operation (but not committed yet)
            self.data[key] = value

            print(f"PUT operation recorded in transaction {txn_id}: {key} = {value}")
            return True

    def get(self, key: str, txn_id: Optional[str] = None) -> Any:
        """Get operation (can be within or outside transaction)."""
        with self.lock:
            if txn_id and not self.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} is not active")

            return self.data.get(key)

    def delete(self, txn_id: str, key: str) -> bool:
        """Delete operation within a transaction."""
        with self.lock:
            if not self.transaction_manager.is_transaction_active(txn_id):
                raise ValueError(f"Transaction {txn_id} not found or not active")

            if key not in self.data:
                return False

            # Record operation for potential rollback
            operation = {
                "type": "delete",
                "key": key,
                "old_value": self.data[key],
                "timestamp": time.time(),
            }

            self.transaction_manager.add_operation(txn_id, operation)

            # Apply operation
            del self.data[key]

            print(f"DELETE operation recorded in transaction {txn_id}: {key}")
            return True

    def commit_transaction(self, txn_id: str) -> bool:
        """Commit transaction using two-phase commit protocol."""
        with self.lock:
            try:
                # Phase 1: Prepare
                self.transaction_manager.prepare_transaction(txn_id)

                # Phase 2: Commit
                self.transaction_manager.commit_transaction(txn_id)

                # Create checkpoint after successful commit
                self.checkpoint_manager.create_checkpoint(
                    self.data, f"post_commit_{txn_id}"
                )

                return True

            except Exception as e:
                print(f"Transaction {txn_id} failed during commit: {e}")
                self.rollback_transaction(txn_id)
                return False

    def rollback_transaction(self, txn_id: str) -> bool:
        """Rollback transaction - implements fault recovery."""
        with self.lock:
            txn = self.transaction_manager.get_transaction(txn_id)
            if not txn:
                print(f"Transaction {txn_id} not found for rollback")
                return False

            print(f"Rolling back transaction {txn_id}")

            # Reverse all operations in reverse order
            for operation in reversed(txn.operations):
                if operation["type"] == "put":
                    if operation["old_value"] is None:
                        # Key was newly created, delete it
                        if operation["key"] in self.data:
                            del self.data[operation["key"]]
                    else:
                        # Key was modified, restore old value
                        self.data[operation["key"]] = operation["old_value"]

                elif operation["type"] == "delete":
                    # Key was deleted, restore it
                    self.data[operation["key"]] = operation["old_value"]

            self.transaction_manager.abort_transaction(txn_id)

            print(f"Transaction {txn_id} rolled back successfully")
            return True

    def rollback_to_checkpoint(self, checkpoint_index: int = -1) -> bool:
        """System-wide rollback to a specific checkpoint."""
        with self.lock:
            try:
                checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_index)
            except ValueError as e:
                print(str(e))
                return False

            # Abort all active transactions
            for txn_id in self.transaction_manager.get_active_transaction_ids():
                print(f"Aborting active transaction {txn_id} due to system rollback")
                self.rollback_transaction(txn_id)

            # Restore data to checkpoint state
            self.data = copy.deepcopy(checkpoint.data)

            print(
                f"System rolled back to checkpoint: {checkpoint.description} "
                f"(timestamp: {checkpoint.timestamp})"
            )

            return True

    def simulate_fault(self) -> None:
        """Simulate a system fault for testing rollback."""
        print("SIMULATING SYSTEM FAULT!")
        print("Rolling back to last checkpoint...")
        self.rollback_to_checkpoint(-1)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        with self.lock:
            return {
                "data_size": len(self.data),
                "active_transactions": self.transaction_manager.get_active_transaction_count(),
                "checkpoints_count": self.checkpoint_manager.get_checkpoint_count(),
                "latest_checkpoint": self.checkpoint_manager.get_latest_checkpoint_description(),
                "current_data": dict(self.data),
            }

    def print_status(self) -> None:
        """Print current system status."""
        status = self.get_system_status()
        print("\n=== SYSTEM STATUS ===")
        print(f"Data entries: {status['data_size']}")
        print(f"Active transactions: {status['active_transactions']}")
        print(f"Checkpoints: {status['checkpoints_count']}")
        print(f"Latest checkpoint: {status['latest_checkpoint']}")
        print(f"Current data: {status['current_data']}")
        print("====================\n")

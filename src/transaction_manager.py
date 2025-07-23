"""Transaction management for ACID compliance."""

import time
import threading
from typing import Dict, Any, Optional
from .models.transaction import Transaction, TransactionState
from .models.checkpoint import Checkpoint


class TransactionManager:
    """Manages ACID transactions with two-phase commit protocol."""

    def __init__(self):
        self.active_transactions: Dict[str, Transaction] = {}
        self.transaction_counter = 0
        self.lock = threading.RLock()

    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        self.transaction_counter += 1
        return f"txn_{self.transaction_counter}_{int(time.time() * 1000)}"

    def begin_transaction(self, checkpoint: Checkpoint) -> str:
        """Begin a new ACID transaction."""
        with self.lock:
            txn_id = self._generate_transaction_id()

            transaction = Transaction(
                id=txn_id,
                state=TransactionState.ACTIVE,
                operations=[],
                timestamp=time.time(),
                checkpoint_before=checkpoint,
            )

            self.active_transactions[txn_id] = transaction
            print(f"Transaction {txn_id} started")
            return txn_id

    def get_transaction(self, txn_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self.active_transactions.get(txn_id)

    def is_transaction_active(self, txn_id: str) -> bool:
        """Check if transaction exists and is active."""
        txn = self.get_transaction(txn_id)
        return txn is not None and txn.state == TransactionState.ACTIVE

    def add_operation(self, txn_id: str, operation: Dict[str, Any]) -> None:
        """Add operation to transaction log."""
        with self.lock:
            if txn_id not in self.active_transactions:
                raise ValueError(f"Transaction {txn_id} not found")

            txn = self.active_transactions[txn_id]
            if txn.state != TransactionState.ACTIVE:
                raise ValueError(f"Transaction {txn_id} is not in active state")

            txn.operations.append(operation)

    def prepare_transaction(self, txn_id: str) -> bool:
        """Phase 1 of two-phase commit: prepare transaction."""
        with self.lock:
            if txn_id not in self.active_transactions:
                raise ValueError(f"Transaction {txn_id} not found")

            txn = self.active_transactions[txn_id]
            print(f"Phase 1: Preparing transaction {txn_id}")
            txn.state = TransactionState.PREPARING

            # Validate transaction
            self._validate_transaction(txn)
            return True

    def commit_transaction(self, txn_id: str) -> bool:
        """Phase 2 of two-phase commit: commit transaction."""
        with self.lock:
            if txn_id not in self.active_transactions:
                raise ValueError(f"Transaction {txn_id} not found")

            txn = self.active_transactions[txn_id]
            print(f"Phase 2: Committing transaction {txn_id}")
            txn.state = TransactionState.COMMITTED

            # Clean up
            del self.active_transactions[txn_id]
            print(f"Transaction {txn_id} committed successfully")
            return True

    def abort_transaction(self, txn_id: str) -> bool:
        """Abort transaction and mark as aborted."""
        with self.lock:
            if txn_id not in self.active_transactions:
                return False

            txn = self.active_transactions[txn_id]
            txn.state = TransactionState.ABORTED
            del self.active_transactions[txn_id]
            return True

    def get_active_transaction_ids(self) -> list:
        """Get list of all active transaction IDs."""
        return list(self.active_transactions.keys())

    def get_active_transaction_count(self) -> int:
        """Get count of active transactions."""
        return len(self.active_transactions)

    def _validate_transaction(self, txn: Transaction) -> bool:
        """Validate transaction before commit (ACID consistency)."""
        for operation in txn.operations:
            if operation["type"] == "put":
                key = operation["key"]
                value = operation["new_value"]

                # Example validation: keys must be strings, values must not be None
                if not isinstance(key, str):
                    raise ValueError(f"Key must be string: {key}")
                if value is None:
                    raise ValueError(f"Value cannot be None for key: {key}")

        return True

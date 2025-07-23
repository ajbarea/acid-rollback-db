"""Unit tests for TransactionManager."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.transaction_manager import TransactionManager
from src.models.transaction import Transaction, TransactionState
from src.models.checkpoint import Checkpoint


# Test fixtures
@pytest.fixture
def transaction_manager():
    """Provide a fresh TransactionManager instance for testing."""
    return TransactionManager()


@pytest.fixture
def sample_checkpoint():
    """Provide a sample checkpoint for testing."""
    return Checkpoint(
        timestamp=time.time(),
        data={"key1": "value1", "key2": "value2"},
        transaction_id="test_txn",
        description="Test checkpoint",
    )


@pytest.fixture
def sample_operations():
    """Provide sample operations for testing."""
    return [
        {
            "type": "put",
            "key": "user_1",
            "new_value": {"name": "Alice", "balance": 1000},
        },
        {"type": "put", "key": "user_2", "new_value": {"name": "Bob", "balance": 500}},
        {"type": "delete", "key": "old_user"},
    ]


class TestTransactionLifecycle:
    """Tests for basic transaction lifecycle operations."""

    def test_begin_transaction_creates_active_transaction(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that begin_transaction creates a new active transaction."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        assert txn_id is not None
        assert txn_id.startswith("txn_")
        assert transaction_manager.is_transaction_active(txn_id)
        assert transaction_manager.get_active_transaction_count() == 1

        txn = transaction_manager.get_transaction(txn_id)
        assert txn.state == TransactionState.ACTIVE
        assert txn.checkpoint_before == sample_checkpoint
        assert txn.operations == []

    def test_begin_transaction_generates_unique_ids(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that each transaction gets a unique ID."""
        txn_id1 = transaction_manager.begin_transaction(sample_checkpoint)
        txn_id2 = transaction_manager.begin_transaction(sample_checkpoint)

        assert txn_id1 != txn_id2
        assert transaction_manager.get_active_transaction_count() == 2

    def test_get_transaction_returns_correct_transaction(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that get_transaction returns the correct transaction."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        txn = transaction_manager.get_transaction(txn_id)
        assert txn is not None
        assert txn.id == txn_id
        assert txn.state == TransactionState.ACTIVE

    def test_get_transaction_returns_none_for_invalid_id(self, transaction_manager):
        """Test that get_transaction returns None for non-existent transaction."""
        txn = transaction_manager.get_transaction("invalid_txn_id")
        assert txn is None

    def test_is_transaction_active_returns_true_for_active_transaction(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that is_transaction_active returns True for active transactions."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        assert transaction_manager.is_transaction_active(txn_id) is True

    def test_is_transaction_active_returns_false_for_invalid_transaction(
        self, transaction_manager
    ):
        """Test that is_transaction_active returns False for non-existent transactions."""
        assert transaction_manager.is_transaction_active("invalid_txn_id") is False

    def test_get_active_transaction_ids_returns_correct_list(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that get_active_transaction_ids returns correct list of IDs."""
        txn_id1 = transaction_manager.begin_transaction(sample_checkpoint)
        txn_id2 = transaction_manager.begin_transaction(sample_checkpoint)

        active_ids = transaction_manager.get_active_transaction_ids()
        assert len(active_ids) == 2
        assert txn_id1 in active_ids
        assert txn_id2 in active_ids

    def test_get_active_transaction_count_returns_correct_count(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that get_active_transaction_count returns correct count."""
        assert transaction_manager.get_active_transaction_count() == 0

        transaction_manager.begin_transaction(sample_checkpoint)
        assert transaction_manager.get_active_transaction_count() == 1

        transaction_manager.begin_transaction(sample_checkpoint)
        assert transaction_manager.get_active_transaction_count() == 2


class TestTransactionOperations:
    """Tests for transaction operation management."""

    def test_add_operation_to_active_transaction(
        self, transaction_manager, sample_checkpoint
    ):
        """Test adding operations to an active transaction."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        operation = {"type": "put", "key": "test_key", "new_value": "test_value"}

        transaction_manager.add_operation(txn_id, operation)

        txn = transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == 1
        assert txn.operations[0] == operation

    def test_add_multiple_operations_to_transaction(
        self, transaction_manager, sample_checkpoint, sample_operations
    ):
        """Test adding multiple operations to a transaction."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        for operation in sample_operations:
            transaction_manager.add_operation(txn_id, operation)

        txn = transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == len(sample_operations)
        assert txn.operations == sample_operations

    def test_add_operation_to_nonexistent_transaction_raises_error(
        self, transaction_manager
    ):
        """Test that adding operation to non-existent transaction raises ValueError."""
        operation = {"type": "put", "key": "test_key", "new_value": "test_value"}

        with pytest.raises(ValueError, match="Transaction invalid_txn not found"):
            transaction_manager.add_operation("invalid_txn", operation)

    def test_add_operation_to_inactive_transaction_raises_error(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that adding operation to inactive transaction raises ValueError."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        operation = {"type": "put", "key": "test_key", "new_value": "test_value"}

        # Manually set transaction to non-active state
        txn = transaction_manager.get_transaction(txn_id)
        txn.state = TransactionState.COMMITTED

        with pytest.raises(
            ValueError, match=f"Transaction {txn_id} is not in active state"
        ):
            transaction_manager.add_operation(txn_id, operation)


class TestTwoPhaseCommitProtocol:
    """Tests for two-phase commit protocol implementation."""

    def test_prepare_transaction_changes_state_to_preparing(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that prepare_transaction changes state to PREPARING."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        result = transaction_manager.prepare_transaction(txn_id)

        assert result is True
        txn = transaction_manager.get_transaction(txn_id)
        assert txn.state == TransactionState.PREPARING

    def test_prepare_nonexistent_transaction_raises_error(self, transaction_manager):
        """Test that preparing non-existent transaction raises ValueError."""
        with pytest.raises(ValueError, match="Transaction invalid_txn not found"):
            transaction_manager.prepare_transaction("invalid_txn")

    def test_commit_transaction_changes_state_and_removes_from_active(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that commit_transaction changes state to COMMITTED and removes from active transactions."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        transaction_manager.prepare_transaction(txn_id)

        result = transaction_manager.commit_transaction(txn_id)

        assert result is True
        assert transaction_manager.get_transaction(txn_id) is None
        assert transaction_manager.get_active_transaction_count() == 0

    def test_commit_nonexistent_transaction_raises_error(self, transaction_manager):
        """Test that committing non-existent transaction raises ValueError."""
        with pytest.raises(ValueError, match="Transaction invalid_txn not found"):
            transaction_manager.commit_transaction("invalid_txn")

    def test_abort_transaction_changes_state_and_removes_from_active(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that abort_transaction changes state to ABORTED and removes from active transactions."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        result = transaction_manager.abort_transaction(txn_id)

        assert result is True
        assert transaction_manager.get_transaction(txn_id) is None
        assert transaction_manager.get_active_transaction_count() == 0

    def test_abort_nonexistent_transaction_returns_false(self, transaction_manager):
        """Test that aborting non-existent transaction returns False."""
        result = transaction_manager.abort_transaction("invalid_txn")
        assert result is False

    def test_complete_two_phase_commit_workflow(
        self, transaction_manager, sample_checkpoint, sample_operations
    ):
        """Test complete two-phase commit workflow from begin to commit."""
        # Begin transaction
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        assert transaction_manager.is_transaction_active(txn_id)

        # Add operations
        for operation in sample_operations:
            transaction_manager.add_operation(txn_id, operation)

        # Phase 1: Prepare
        prepare_result = transaction_manager.prepare_transaction(txn_id)
        assert prepare_result is True
        txn = transaction_manager.get_transaction(txn_id)
        assert txn.state == TransactionState.PREPARING

        # Phase 2: Commit
        commit_result = transaction_manager.commit_transaction(txn_id)
        assert commit_result is True
        assert transaction_manager.get_transaction(txn_id) is None


class TestTransactionValidation:
    """Tests for transaction validation logic."""

    def test_validate_transaction_with_valid_operations(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that validation passes for valid operations."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        valid_operation = {
            "type": "put",
            "key": "valid_key",
            "new_value": "valid_value",
        }

        transaction_manager.add_operation(txn_id, valid_operation)

        # Should not raise exception
        result = transaction_manager.prepare_transaction(txn_id)
        assert result is True

    def test_validate_transaction_with_invalid_key_type_raises_error(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that validation fails for non-string keys."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        invalid_operation = {"type": "put", "key": 123, "new_value": "valid_value"}

        transaction_manager.add_operation(txn_id, invalid_operation)

        with pytest.raises(ValueError, match="Key must be string: 123"):
            transaction_manager.prepare_transaction(txn_id)

    def test_validate_transaction_with_none_value_raises_error(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that validation fails for None values."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        invalid_operation = {"type": "put", "key": "valid_key", "new_value": None}

        transaction_manager.add_operation(txn_id, invalid_operation)

        with pytest.raises(ValueError, match="Value cannot be None for key: valid_key"):
            transaction_manager.prepare_transaction(txn_id)

    def test_validate_transaction_with_mixed_operations(
        self, transaction_manager, sample_checkpoint
    ):
        """Test validation with mix of valid and invalid operations."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        # Add valid operation
        valid_operation = {
            "type": "put",
            "key": "valid_key",
            "new_value": "valid_value",
        }
        transaction_manager.add_operation(txn_id, valid_operation)

        # Add invalid operation
        invalid_operation = {"type": "put", "key": "invalid_key", "new_value": None}
        transaction_manager.add_operation(txn_id, invalid_operation)

        with pytest.raises(
            ValueError, match="Value cannot be None for key: invalid_key"
        ):
            transaction_manager.prepare_transaction(txn_id)


class TestConcurrentTransactions:
    """Tests for concurrent transaction handling and thread safety."""

    def test_concurrent_transaction_creation(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that multiple transactions can be created concurrently."""
        num_threads = 10
        transaction_ids = []

        def create_transaction():
            txn_id = transaction_manager.begin_transaction(sample_checkpoint)
            return txn_id

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_transaction) for _ in range(num_threads)]
            transaction_ids = [future.result() for future in as_completed(futures)]

        # All transactions should be unique and active
        assert len(transaction_ids) == num_threads
        assert len(set(transaction_ids)) == num_threads  # All unique
        assert transaction_manager.get_active_transaction_count() == num_threads

    def test_concurrent_operation_addition(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that operations can be added to transactions concurrently."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)
        num_operations = 20

        def add_operation(index):
            operation = {
                "type": "put",
                "key": f"key_{index}",
                "new_value": f"value_{index}",
            }
            transaction_manager.add_operation(txn_id, operation)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_operation, i) for i in range(num_operations)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        txn = transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == num_operations

    def test_concurrent_transaction_lifecycle_operations(
        self, transaction_manager, sample_checkpoint
    ):
        """Test concurrent lifecycle operations on different transactions."""
        num_transactions = 5
        transaction_ids = []

        # Create multiple transactions
        for _ in range(num_transactions):
            txn_id = transaction_manager.begin_transaction(sample_checkpoint)
            transaction_ids.append(txn_id)

        def process_transaction(txn_id):
            # Add operation
            operation = {
                "type": "put",
                "key": f"key_{txn_id}",
                "new_value": f"value_{txn_id}",
            }
            transaction_manager.add_operation(txn_id, operation)

            # Prepare and commit
            transaction_manager.prepare_transaction(txn_id)
            transaction_manager.commit_transaction(txn_id)

        with ThreadPoolExecutor(max_workers=num_transactions) as executor:
            futures = [
                executor.submit(process_transaction, txn_id)
                for txn_id in transaction_ids
            ]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # All transactions should be committed and removed
        assert transaction_manager.get_active_transaction_count() == 0

    def test_thread_safety_with_mixed_operations(
        self, transaction_manager, sample_checkpoint
    ):
        """Test thread safety with mixed transaction operations."""
        results = []

        def mixed_operations():
            try:
                # Create transaction
                txn_id = transaction_manager.begin_transaction(sample_checkpoint)

                # Add operation
                operation = {
                    "type": "put",
                    "key": "test_key",
                    "new_value": "test_value",
                }
                transaction_manager.add_operation(txn_id, operation)

                # Check if active
                is_active = transaction_manager.is_transaction_active(txn_id)

                # Abort transaction
                abort_result = transaction_manager.abort_transaction(txn_id)

                return {
                    "success": True,
                    "txn_id": txn_id,
                    "is_active": is_active,
                    "abort_result": abort_result,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_operations) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]

        # All operations should succeed
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == 10

        # All transactions should be cleaned up
        assert transaction_manager.get_active_transaction_count() == 0

    @patch("time.time")
    def test_transaction_id_generation_thread_safety(
        self, mock_time, transaction_manager, sample_checkpoint
    ):
        """Test that transaction ID generation is thread-safe."""
        mock_time.return_value = 1234567890.0

        def create_transaction():
            return transaction_manager.begin_transaction(sample_checkpoint)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(create_transaction) for _ in range(50)]
            transaction_ids = [future.result() for future in as_completed(futures)]

        # All transaction IDs should be unique despite same timestamp
        assert len(set(transaction_ids)) == 50


class TestTransactionManagerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_transaction_manager_initialization(self):
        """Test that TransactionManager initializes correctly."""
        tm = TransactionManager()
        assert tm.active_transactions == {}
        assert tm.transaction_counter == 0
        assert tm.lock is not None
        assert tm.get_active_transaction_count() == 0

    def test_transaction_state_transitions(
        self, transaction_manager, sample_checkpoint
    ):
        """Test valid transaction state transitions."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        # ACTIVE -> PREPARING
        txn = transaction_manager.get_transaction(txn_id)
        assert txn.state == TransactionState.ACTIVE

        transaction_manager.prepare_transaction(txn_id)
        txn = transaction_manager.get_transaction(txn_id)
        assert txn.state == TransactionState.PREPARING

        # PREPARING -> COMMITTED (and removed)
        transaction_manager.commit_transaction(txn_id)
        assert transaction_manager.get_transaction(txn_id) is None

    def test_transaction_abort_from_different_states(
        self, transaction_manager, sample_checkpoint
    ):
        """Test that transactions can be aborted from different states."""
        # Test abort from ACTIVE state
        txn_id1 = transaction_manager.begin_transaction(sample_checkpoint)
        result1 = transaction_manager.abort_transaction(txn_id1)
        assert result1 is True

        # Test abort from PREPARING state
        txn_id2 = transaction_manager.begin_transaction(sample_checkpoint)
        transaction_manager.prepare_transaction(txn_id2)
        result2 = transaction_manager.abort_transaction(txn_id2)
        assert result2 is True

    def test_empty_transaction_validation(self, transaction_manager, sample_checkpoint):
        """Test validation of transaction with no operations."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        # Should be able to prepare and commit empty transaction
        result = transaction_manager.prepare_transaction(txn_id)
        assert result is True

        result = transaction_manager.commit_transaction(txn_id)
        assert result is True

    @patch("builtins.print")
    def test_transaction_logging_output(
        self, mock_print, transaction_manager, sample_checkpoint
    ):
        """Test that transaction operations produce expected log output."""
        txn_id = transaction_manager.begin_transaction(sample_checkpoint)

        # Check begin transaction log
        mock_print.assert_called_with(f"Transaction {txn_id} started")

        # Prepare transaction
        transaction_manager.prepare_transaction(txn_id)
        mock_print.assert_called_with(f"Phase 1: Preparing transaction {txn_id}")

        # Commit transaction
        transaction_manager.commit_transaction(txn_id)
        expected_calls = [
            f"Phase 2: Committing transaction {txn_id}",
            f"Transaction {txn_id} committed successfully",
        ]

        # Check that both commit messages were printed
        actual_calls = [call.args[0] for call in mock_print.call_args_list[-2:]]
        assert actual_calls == expected_calls

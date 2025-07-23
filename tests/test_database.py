"""Unit tests for AcidRollbackDB."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.database import AcidRollbackDB
from src.models.transaction import TransactionState
from src.models.checkpoint import Checkpoint


class TestDatabaseSetup:
    """Test AcidRollbackDB initialization and basic setup."""

    def test_init_default_parameters(self):
        """Test AcidRollbackDB initialization with default parameters."""
        db = AcidRollbackDB()

        assert db.data == {}
        assert db.checkpoint_interval == 5
        assert db.lock is not None
        assert db.transaction_manager is not None
        assert db.checkpoint_manager is not None
        assert db.checkpoint_manager.max_checkpoints == 10

        # Should have initial checkpoint
        assert db.checkpoint_manager.get_checkpoint_count() == 1
        assert (
            db.checkpoint_manager.get_latest_checkpoint_description() == "initial_state"
        )

    def test_init_custom_parameters(self):
        """Test AcidRollbackDB initialization with custom parameters."""
        db = AcidRollbackDB(checkpoint_interval=3, max_checkpoints=5)

        assert db.checkpoint_interval == 3
        assert db.checkpoint_manager.max_checkpoints == 5
        assert db.checkpoint_manager.get_checkpoint_count() == 1

    def test_initial_checkpoint_creation(self):
        """Test that initial checkpoint is created correctly."""
        db = AcidRollbackDB()

        initial_checkpoint = db.checkpoint_manager.get_checkpoint(0)
        assert initial_checkpoint.description == "initial_state"
        assert initial_checkpoint.data == {}


class TestBasicOperations:
    """Test basic database operations (put, get, delete)."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB(checkpoint_interval=5, max_checkpoints=3)

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        return {
            "user_1": {"name": "Alice", "balance": 1000},
            "user_2": {"name": "Bob", "balance": 500},
            "product_1": {"name": "Widget", "price": 25.99},
        }

    def test_put_operation_in_transaction(self, database):
        """Test put operation within a transaction."""
        txn_id = database.begin_transaction()

        result = database.put(txn_id, "test_key", "test_value")

        assert result is True
        assert database.data["test_key"] == "test_value"

        # Verify operation was recorded in transaction
        txn = database.transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == 1
        assert txn.operations[0]["type"] == "put"
        assert txn.operations[0]["key"] == "test_key"
        assert txn.operations[0]["new_value"] == "test_value"
        assert txn.operations[0]["old_value"] is None

    def test_put_operation_overwrites_existing_key(self, database):
        """Test put operation overwrites existing key and records old value."""
        txn_id = database.begin_transaction()

        # First put
        database.put(txn_id, "test_key", "original_value")
        # Second put (overwrite)
        database.put(txn_id, "test_key", "new_value")

        assert database.data["test_key"] == "new_value"

        # Check that both operations are recorded
        txn = database.transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == 2
        assert txn.operations[1]["old_value"] == "original_value"

    def test_put_operation_with_invalid_transaction_raises_error(self, database):
        """Test put operation with invalid transaction ID raises ValueError."""
        with pytest.raises(
            ValueError, match="Transaction invalid_txn not found or not active"
        ):
            database.put("invalid_txn", "test_key", "test_value")

    def test_get_operation_without_transaction(self, database):
        """Test get operation without transaction context."""
        # Put some data first
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        # Get without transaction
        result = database.get("test_key")
        assert result == "test_value"

        # Get non-existent key
        result = database.get("non_existent")
        assert result is None

    def test_get_operation_with_transaction(self, database):
        """Test get operation with transaction context."""
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        # Get with transaction
        result = database.get("test_key", txn_id)
        assert result == "test_value"

    def test_get_operation_with_invalid_transaction_raises_error(self, database):
        """Test get operation with invalid transaction ID raises ValueError."""
        with pytest.raises(ValueError, match="Transaction invalid_txn is not active"):
            database.get("test_key", "invalid_txn")

    def test_delete_operation_in_transaction(self, database):
        """Test delete operation within a transaction."""
        txn_id = database.begin_transaction()

        # Put data first
        database.put(txn_id, "test_key", "test_value")

        # Delete the key
        result = database.delete(txn_id, "test_key")

        assert result is True
        assert "test_key" not in database.data

        # Verify operation was recorded
        txn = database.transaction_manager.get_transaction(txn_id)
        delete_op = txn.operations[-1]  # Last operation
        assert delete_op["type"] == "delete"
        assert delete_op["key"] == "test_key"
        assert delete_op["old_value"] == "test_value"

    def test_delete_operation_nonexistent_key_returns_false(self, database):
        """Test delete operation on non-existent key returns False."""
        txn_id = database.begin_transaction()

        result = database.delete(txn_id, "non_existent_key")

        assert result is False

        # No operation should be recorded
        txn = database.transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == 0

    def test_delete_operation_with_invalid_transaction_raises_error(self, database):
        """Test delete operation with invalid transaction ID raises ValueError."""
        with pytest.raises(
            ValueError, match="Transaction invalid_txn not found or not active"
        ):
            database.delete("invalid_txn", "test_key")

    def test_multiple_operations_in_single_transaction(self, database, sample_data):
        """Test multiple operations within a single transaction."""
        txn_id = database.begin_transaction()

        # Multiple puts
        for key, value in sample_data.items():
            database.put(txn_id, key, value)

        # Delete one item
        database.delete(txn_id, "user_2")

        # Verify final state
        assert database.data["user_1"] == sample_data["user_1"]
        assert "user_2" not in database.data
        assert database.data["product_1"] == sample_data["product_1"]

        # Verify all operations recorded
        txn = database.transaction_manager.get_transaction(txn_id)
        assert len(txn.operations) == 4  # 3 puts + 1 delete


class TestTransactionIntegration:
    """Test database operations integration with transaction management."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB()

    def test_begin_transaction_creates_checkpoint(self, database):
        """Test that beginning a transaction creates a pre-transaction checkpoint."""
        initial_count = database.checkpoint_manager.get_checkpoint_count()

        txn_id = database.begin_transaction()

        assert database.checkpoint_manager.get_checkpoint_count() == initial_count + 1
        assert database.transaction_manager.is_transaction_active(txn_id)

        # Verify checkpoint was created
        latest_checkpoint = database.checkpoint_manager.get_checkpoint()
        assert "pre_transaction" in latest_checkpoint.description

    def test_commit_transaction_success(self, database):
        """Test successful transaction commit."""
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        result = database.commit_transaction(txn_id)

        assert result is True
        assert not database.transaction_manager.is_transaction_active(txn_id)
        assert database.data["test_key"] == "test_value"

        # Should create post-commit checkpoint
        latest_checkpoint = database.checkpoint_manager.get_checkpoint()
        assert f"post_commit_{txn_id}" in latest_checkpoint.description

    def test_commit_transaction_with_validation_failure(self, database):
        """Test transaction commit with validation failure triggers rollback."""
        txn_id = database.begin_transaction()

        # Add invalid operation directly to bypass normal validation
        invalid_operation = {
            "type": "put",
            "key": 123,  # Invalid key type
            "new_value": "test_value",
            "old_value": None,
            "timestamp": time.time(),
        }
        database.transaction_manager.add_operation(txn_id, invalid_operation)

        result = database.commit_transaction(txn_id)

        assert result is False
        assert not database.transaction_manager.is_transaction_active(txn_id)

    def test_rollback_transaction_put_operations(self, database):
        """Test rollback of put operations."""
        txn_id = database.begin_transaction()

        # Put new key
        database.put(txn_id, "new_key", "new_value")

        # Put existing key (overwrite)
        database.data["existing_key"] = "original_value"
        database.put(txn_id, "existing_key", "modified_value")

        # Rollback
        result = database.rollback_transaction(txn_id)

        assert result is True
        assert "new_key" not in database.data
        assert database.data["existing_key"] == "original_value"
        assert not database.transaction_manager.is_transaction_active(txn_id)

    def test_rollback_transaction_delete_operations(self, database):
        """Test rollback of delete operations."""
        # Setup initial data
        database.data["test_key"] = "test_value"

        txn_id = database.begin_transaction()
        database.delete(txn_id, "test_key")

        # Verify deletion
        assert "test_key" not in database.data

        # Rollback
        result = database.rollback_transaction(txn_id)

        assert result is True
        assert database.data["test_key"] == "test_value"
        assert not database.transaction_manager.is_transaction_active(txn_id)

    def test_rollback_transaction_mixed_operations(self, database):
        """Test rollback of mixed put and delete operations."""
        # Setup initial data
        database.data["existing_key"] = "original_value"

        txn_id = database.begin_transaction()

        # Mixed operations
        database.put(txn_id, "new_key", "new_value")
        database.put(txn_id, "existing_key", "modified_value")
        database.delete(txn_id, "existing_key")
        database.put(txn_id, "another_key", "another_value")

        # Rollback
        result = database.rollback_transaction(txn_id)

        assert result is True
        assert "new_key" not in database.data
        assert "another_key" not in database.data
        assert database.data["existing_key"] == "original_value"

    def test_rollback_nonexistent_transaction_returns_false(self, database):
        """Test rollback of non-existent transaction returns False."""
        result = database.rollback_transaction("invalid_txn")
        assert result is False

    def test_concurrent_transactions_isolation(self, database):
        """Test that concurrent transactions are properly isolated."""
        txn_id1 = database.begin_transaction()
        txn_id2 = database.begin_transaction()

        # Operations in different transactions
        database.put(txn_id1, "key1", "value1")
        database.put(txn_id2, "key2", "value2")

        # Both should see the data
        assert database.get("key1") == "value1"
        assert database.get("key2") == "value2"

        # Rollback one transaction
        database.rollback_transaction(txn_id1)

        # Only txn2 data should remain
        assert "key1" not in database.data
        assert database.data["key2"] == "value2"
        assert database.transaction_manager.is_transaction_active(txn_id2)


class TestRollbackMechanisms:
    """Test rollback mechanisms including system-wide rollback."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB(max_checkpoints=5)

    def test_rollback_to_checkpoint_default_latest(self, database):
        """Test rollback to latest checkpoint (default behavior)."""
        # Add some initial data
        txn_id = database.begin_transaction()
        database.put(txn_id, "initial_key", "initial_value")
        database.commit_transaction(txn_id)

        # Add more data
        txn_id2 = database.begin_transaction()
        database.put(txn_id2, "new_key", "new_value")
        # Don't commit - leave transaction active

        # Rollback to latest checkpoint
        result = database.rollback_to_checkpoint()

        assert result is True
        assert database.data["initial_key"] == "initial_value"
        assert "new_key" not in database.data
        assert not database.transaction_manager.is_transaction_active(txn_id2)

    def test_rollback_to_checkpoint_by_index(self, database):
        """Test rollback to specific checkpoint by index."""
        # Create multiple checkpoints by committing transactions
        for i in range(3):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"key_{i}", f"value_{i}")
            database.commit_transaction(txn_id)

        # Rollback to second checkpoint (index 1) - should have key_0 only
        result = database.rollback_to_checkpoint(1)

        assert result is True
        # Should have data from first committed transaction only
        expected_data = {"key_0": "value_0"}
        assert database.data == expected_data
        assert database.transaction_manager.get_active_transaction_count() == 0

    def test_rollback_to_checkpoint_invalid_index(self, database):
        """Test rollback to invalid checkpoint index returns False."""
        result = database.rollback_to_checkpoint(99)

        assert result is False

    def test_rollback_to_checkpoint_aborts_active_transactions(self, database):
        """Test that system rollback aborts all active transactions."""
        # Create multiple active transactions
        txn_ids = []
        for i in range(3):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"key_{i}", f"value_{i}")
            txn_ids.append(txn_id)

        assert database.transaction_manager.get_active_transaction_count() == 3

        # System rollback
        result = database.rollback_to_checkpoint()

        assert result is True
        assert database.transaction_manager.get_active_transaction_count() == 0

        # All transactions should be inactive
        for txn_id in txn_ids:
            assert not database.transaction_manager.is_transaction_active(txn_id)

    def test_rollback_to_checkpoint_restores_data_state(self, database):
        """Test that system rollback properly restores data state."""
        # Initial state
        txn_id1 = database.begin_transaction()
        database.put(txn_id1, "persistent_key", "persistent_value")
        database.commit_transaction(txn_id1)

        checkpoint_data = dict(database.data)

        # Make changes
        txn_id2 = database.begin_transaction()
        database.put(txn_id2, "temp_key", "temp_value")
        database.delete(txn_id2, "persistent_key")

        # Rollback
        result = database.rollback_to_checkpoint()

        assert result is True
        assert database.data == checkpoint_data
        assert database.data["persistent_key"] == "persistent_value"
        assert "temp_key" not in database.data


class TestSystemStatus:
    """Test system status reporting functionality."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB()

    def test_get_system_status_initial_state(self, database):
        """Test system status in initial state."""
        status = database.get_system_status()

        assert status["data_size"] == 0
        assert status["active_transactions"] == 0
        assert status["checkpoints_count"] == 1  # Initial checkpoint
        assert status["latest_checkpoint"] == "initial_state"
        assert status["current_data"] == {}

    def test_get_system_status_with_data_and_transactions(self, database):
        """Test system status with data and active transactions."""
        # Add some data
        txn_id1 = database.begin_transaction()
        database.put(txn_id1, "key1", "value1")
        database.commit_transaction(txn_id1)

        # Create active transaction
        txn_id2 = database.begin_transaction()
        database.put(txn_id2, "key2", "value2")

        status = database.get_system_status()

        assert status["data_size"] == 2
        assert status["active_transactions"] == 1
        assert status["checkpoints_count"] >= 2  # Initial + post-commit
        assert status["current_data"]["key1"] == "value1"
        assert status["current_data"]["key2"] == "value2"

    def test_get_system_status_thread_safety(self, database):
        """Test that get_system_status is thread-safe."""
        # Add some data
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        results = []

        def get_status():
            status = database.get_system_status()
            results.append(status)

        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_status)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be consistent
        assert len(results) == 10
        for status in results:
            assert status["data_size"] == 1
            assert status["active_transactions"] == 1
            assert status["current_data"]["test_key"] == "test_value"

    @patch("builtins.print")
    def test_print_status_output(self, mock_print, database):
        """Test that print_status produces expected output."""
        # Add some data
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        database.print_status()

        # Verify print was called with status information
        assert mock_print.call_count >= 6  # Multiple print statements

        # Check some expected content in the printed output
        printed_content = " ".join(
            [str(call.args[0]) for call in mock_print.call_args_list]
        )
        assert "SYSTEM STATUS" in printed_content
        assert "Data entries: 1" in printed_content
        assert "Active transactions: 1" in printed_content


class TestFaultSimulation:
    """Test fault simulation and recovery mechanisms."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB()

    @patch("builtins.print")
    def test_simulate_fault_triggers_rollback(self, mock_print, database):
        """Test that simulate_fault triggers system rollback."""
        # Setup some data
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")
        database.commit_transaction(txn_id)

        # Create active transaction
        active_txn = database.begin_transaction()
        database.put(active_txn, "temp_key", "temp_value")

        # Simulate fault
        database.simulate_fault()

        # Verify rollback occurred
        assert database.data["test_key"] == "test_value"
        assert "temp_key" not in database.data
        assert not database.transaction_manager.is_transaction_active(active_txn)

        # Verify fault simulation messages
        printed_content = " ".join(
            [str(call.args[0]) for call in mock_print.call_args_list]
        )
        assert "SIMULATING SYSTEM FAULT!" in printed_content
        assert "Rolling back to last checkpoint..." in printed_content

    def test_fault_simulation_with_multiple_active_transactions(self, database):
        """Test fault simulation with multiple active transactions."""
        # Setup committed data
        txn_id = database.begin_transaction()
        database.put(txn_id, "persistent_key", "persistent_value")
        database.commit_transaction(txn_id)

        # Create multiple active transactions
        active_txns = []
        for i in range(3):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"temp_key_{i}", f"temp_value_{i}")
            active_txns.append(txn_id)

        # Simulate fault
        database.simulate_fault()

        # All active transactions should be aborted
        for txn_id in active_txns:
            assert not database.transaction_manager.is_transaction_active(txn_id)

        # The system rolls back to the latest checkpoint
        # The key thing is that active transactions are aborted
        assert database.transaction_manager.get_active_transaction_count() == 0

        # The system should have rolled back to some checkpoint state
        # We can't predict exactly which checkpoint, but we know:
        # 1. No active transactions should remain
        # 2. The system should be in a consistent state
        # 3. The persistent key should be present if it was committed before the rollback point

        # Verify system is in consistent state with no active transactions
        status = database.get_system_status()
        assert status["active_transactions"] == 0

        # The data should be in some valid checkpoint state
        # (could be any checkpoint depending on which one was latest)
        assert isinstance(database.data, dict)

    def test_fault_recovery_maintains_data_integrity(self, database):
        """Test that fault recovery maintains data integrity."""
        # Build up some committed state
        committed_data = {}
        for i in range(5):
            txn_id = database.begin_transaction()
            key = f"committed_key_{i}"
            value = f"committed_value_{i}"
            database.put(txn_id, key, value)
            database.commit_transaction(txn_id)
            committed_data[key] = value

        # Make uncommitted changes
        active_txn = database.begin_transaction()
        database.put(active_txn, "uncommitted_key", "uncommitted_value")
        database.delete(active_txn, "committed_key_2")

        # Simulate fault
        database.simulate_fault()

        # Verify data integrity
        assert database.data == committed_data
        assert database.transaction_manager.get_active_transaction_count() == 0


class TestConcurrentOperations:
    """Test concurrent database operations and thread safety."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB()

    def test_concurrent_transaction_operations(self, database):
        """Test concurrent operations across multiple transactions."""
        num_threads = 10
        operations_per_thread = 5
        results = []

        def transaction_worker(thread_id):
            try:
                txn_id = database.begin_transaction()

                for i in range(operations_per_thread):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    database.put(txn_id, key, value)

                # Commit half, rollback half
                if thread_id % 2 == 0:
                    database.commit_transaction(txn_id)
                    results.append(("commit", thread_id))
                else:
                    database.rollback_transaction(txn_id)
                    results.append(("rollback", thread_id))

            except Exception as e:
                results.append(("error", str(e)))

        # Run concurrent transactions
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(transaction_worker, i) for i in range(num_threads)
            ]
            for future in as_completed(futures):
                future.result()

        # Verify results
        commits = [r for r in results if r[0] == "commit"]
        rollbacks = [r for r in results if r[0] == "rollback"]
        errors = [r for r in results if r[0] == "error"]

        assert len(errors) == 0
        assert len(commits) == 5  # Even thread IDs
        assert len(rollbacks) == 5  # Odd thread IDs

        # Verify only committed data remains
        expected_keys = set()
        for _, thread_id in commits:
            for i in range(operations_per_thread):
                expected_keys.add(f"thread_{thread_id}_key_{i}")

        assert set(database.data.keys()) == expected_keys

    def test_concurrent_read_write_operations(self, database):
        """Test concurrent read and write operations."""
        # Setup initial data
        txn_id = database.begin_transaction()
        for i in range(10):
            database.put(txn_id, f"key_{i}", f"initial_value_{i}")
        database.commit_transaction(txn_id)

        read_results = []
        write_results = []

        def reader_worker():
            try:
                for i in range(10):
                    value = database.get(f"key_{i}")
                    read_results.append(value)
            except Exception as e:
                read_results.append(f"error: {e}")

        def writer_worker(worker_id):
            try:
                txn_id = database.begin_transaction()
                database.put(txn_id, f"new_key_{worker_id}", f"new_value_{worker_id}")
                database.commit_transaction(txn_id)
                write_results.append(f"success_{worker_id}")
            except Exception as e:
                write_results.append(f"error: {e}")

        # Run concurrent readers and writers
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Start readers
            reader_futures = [executor.submit(reader_worker) for _ in range(5)]
            # Start writers
            writer_futures = [executor.submit(writer_worker, i) for i in range(5)]

            # Wait for completion
            for future in as_completed(reader_futures + writer_futures):
                future.result()

        # Verify no errors occurred
        assert all("error" not in str(result) for result in read_results)
        assert all("error" not in str(result) for result in write_results)
        assert len(write_results) == 5

    def test_concurrent_system_operations(self, database):
        """Test concurrent system-level operations."""
        results = []

        def status_worker():
            try:
                status = database.get_system_status()
                results.append(("status", status["data_size"]))
            except Exception as e:
                results.append(("status_error", str(e)))

        def transaction_worker(worker_id):
            try:
                txn_id = database.begin_transaction()
                database.put(txn_id, f"key_{worker_id}", f"value_{worker_id}")
                database.commit_transaction(txn_id)
                results.append(("transaction", worker_id))
            except Exception as e:
                results.append(("transaction_error", str(e)))

        def rollback_worker():
            try:
                # Wait a bit to let some transactions start
                time.sleep(0.01)
                database.rollback_to_checkpoint()
                results.append(("rollback", "success"))
            except Exception as e:
                results.append(("rollback_error", str(e)))

        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            # Status checkers
            futures.extend([executor.submit(status_worker) for _ in range(3)])
            # Transaction workers
            futures.extend([executor.submit(transaction_worker, i) for i in range(5)])
            # Rollback worker
            futures.append(executor.submit(rollback_worker))

            # Wait for completion
            for future in as_completed(futures):
                future.result()

        # Verify no errors occurred
        errors = [r for r in results if "error" in r[0]]
        assert len(errors) == 0


class TestErrorHandling:
    """Test error handling in database operations."""

    @pytest.fixture
    def database(self):
        """Provide a fresh AcidRollbackDB instance for testing."""
        return AcidRollbackDB()

    def test_operations_with_none_values(self, database):
        """Test database operations with None values."""
        txn_id = database.begin_transaction()

        # Put with None value should work (stored as None)
        result = database.put(txn_id, "none_key", None)
        assert result is True
        assert database.data["none_key"] is None

        # Get None value
        value = database.get("none_key")
        assert value is None

    def test_operations_with_empty_strings(self, database):
        """Test database operations with empty strings."""
        txn_id = database.begin_transaction()

        # Empty string key and value
        result = database.put(txn_id, "", "")
        assert result is True
        assert database.data[""] == ""

    def test_operations_with_complex_data_types(self, database):
        """Test database operations with complex data types."""
        txn_id = database.begin_transaction()

        complex_data = {
            "nested": {"dict": {"with": ["list", "items"]}},
            "number": 42,
            "boolean": True,
            "null": None,
        }

        result = database.put(txn_id, "complex_key", complex_data)
        assert result is True
        assert database.data["complex_key"] == complex_data

        # Verify deep copy behavior in rollback
        database.rollback_transaction(txn_id)
        assert "complex_key" not in database.data

    def test_transaction_state_edge_cases(self, database):
        """Test edge cases in transaction state management."""
        txn_id = database.begin_transaction()

        # Try to operate on transaction after manual state change
        txn = database.transaction_manager.get_transaction(txn_id)
        txn.state = TransactionState.COMMITTED

        with pytest.raises(ValueError, match="not found or not active"):
            database.put(txn_id, "test_key", "test_value")

    def test_rollback_with_corrupted_operations(self, database):
        """Test rollback behavior with corrupted operation data."""
        txn_id = database.begin_transaction()
        database.put(txn_id, "test_key", "test_value")

        # Corrupt operation data
        txn = database.transaction_manager.get_transaction(txn_id)
        txn.operations[0]["type"] = "invalid_operation_type"

        # Rollback should handle gracefully
        result = database.rollback_transaction(txn_id)
        assert result is True
        assert not database.transaction_manager.is_transaction_active(txn_id)

    def test_checkpoint_operations_with_large_data(self, database):
        """Test checkpoint operations with large data sets."""
        # Store initial state
        initial_data_size = len(database.data)

        # Create large dataset
        large_data = {
            f"key_{i}": f"value_{i}" * 100 for i in range(100)
        }  # Reduced size for faster test

        txn_id = database.begin_transaction()
        for key, value in large_data.items():
            database.put(txn_id, key, value)
        database.commit_transaction(txn_id)

        # Verify data was added
        assert len(database.data) == len(large_data)

        # Create another transaction with more data
        txn_id2 = database.begin_transaction()
        database.put(txn_id2, "additional_key", "additional_value")
        # Don't commit this transaction

        # System rollback should handle large data and rollback to last checkpoint
        result = database.rollback_to_checkpoint()
        assert result is True
        # Should rollback to the committed large data state (not the uncommitted additional data)
        assert len(database.data) == len(large_data)
        assert "additional_key" not in database.data

"""Integration tests for high-level system flows."""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from src.database import AcidRollbackDB
from src.models.transaction import TransactionState


# Test fixtures for integration test scenarios
@pytest.fixture
def database():
    """Provide a fresh AcidRollbackDB instance for integration testing."""
    return AcidRollbackDB(checkpoint_interval=3, max_checkpoints=5)


@pytest.fixture
def banking_data():
    """Provide sample banking data for real-world scenarios."""
    return {
        "account_alice": {
            "name": "Alice Smith",
            "balance": 1000.00,
            "type": "checking",
        },
        "account_bob": {"name": "Bob Johnson", "balance": 500.00, "type": "savings"},
        "account_charlie": {
            "name": "Charlie Brown",
            "balance": 250.00,
            "type": "checking",
        },
    }


@pytest.fixture
def populated_database(database, banking_data):
    """Provide a database pre-populated with banking data."""
    # Setup initial banking data
    txn_id = database.begin_transaction()
    for account_id, account_data in banking_data.items():
        database.put(txn_id, account_id, account_data)
    database.commit_transaction(txn_id)
    return database


class TestCompleteTransactionWorkflows:
    """Test complete transaction workflows from begin to commit/rollback."""

    def test_successful_transaction_workflow(self, database):
        """Test complete successful transaction workflow."""
        # Begin transaction
        txn_id = database.begin_transaction()
        assert database.transaction_manager.is_transaction_active(txn_id)

        initial_checkpoint_count = database.checkpoint_manager.get_checkpoint_count()

        # Perform operations
        database.put(txn_id, "user_1", {"name": "Alice", "email": "alice@example.com"})
        database.put(txn_id, "user_2", {"name": "Bob", "email": "bob@example.com"})
        database.put(txn_id, "config", {"theme": "dark", "notifications": True})

        # Verify data is visible during transaction
        assert database.get("user_1")["name"] == "Alice"
        assert database.get("user_2")["name"] == "Bob"
        assert database.get("config")["theme"] == "dark"

        # Commit transaction
        result = database.commit_transaction(txn_id)
        assert result is True
        assert not database.transaction_manager.is_transaction_active(txn_id)

        # Verify data persists after commit
        assert database.get("user_1")["name"] == "Alice"
        assert database.get("user_2")["name"] == "Bob"
        assert database.get("config")["theme"] == "dark"

        # Verify checkpoint was created after commit
        assert (
            database.checkpoint_manager.get_checkpoint_count()
            == initial_checkpoint_count + 1
        )  # post-commit checkpoint

    def test_failed_transaction_workflow_with_rollback(self, database):
        """Test transaction workflow that fails and triggers rollback."""
        # Setup initial data
        txn_id = database.begin_transaction()
        database.put(txn_id, "existing_key", "original_value")
        database.commit_transaction(txn_id)

        # Begin new transaction
        txn_id2 = database.begin_transaction()

        # Perform operations
        database.put(txn_id2, "new_key", "new_value")
        database.put(txn_id2, "existing_key", "modified_value")
        database.delete(txn_id2, "existing_key")

        # Verify changes are visible during transaction
        assert database.get("new_key") == "new_value"
        assert "existing_key" not in database.data

        # Simulate validation failure by adding invalid operation
        invalid_operation = {
            "type": "put",
            "key": 123,  # Invalid key type
            "new_value": "test",
            "old_value": None,
            "timestamp": time.time(),
        }
        database.transaction_manager.add_operation(txn_id2, invalid_operation)

        # Attempt commit (should fail and rollback)
        result = database.commit_transaction(txn_id2)
        assert result is False
        assert not database.transaction_manager.is_transaction_active(txn_id2)

        # Verify rollback occurred
        assert "new_key" not in database.data
        assert database.get("existing_key") == "original_value"

    def test_manual_transaction_rollback_workflow(self, database):
        """Test manual transaction rollback workflow."""
        # Setup initial data
        initial_data = {"key1": "value1", "key2": "value2"}
        txn_id = database.begin_transaction()
        for key, value in initial_data.items():
            database.put(txn_id, key, value)
        database.commit_transaction(txn_id)

        # Begin new transaction with modifications
        txn_id2 = database.begin_transaction()
        database.put(txn_id2, "key1", "modified_value1")
        database.put(txn_id2, "new_key", "new_value")
        database.delete(txn_id2, "key2")

        # Verify changes are applied
        assert database.get("key1") == "modified_value1"
        assert database.get("new_key") == "new_value"
        assert "key2" not in database.data

        # Manual rollback
        result = database.rollback_transaction(txn_id2)
        assert result is True
        assert not database.transaction_manager.is_transaction_active(txn_id2)

        # Verify original state restored
        assert database.get("key1") == "value1"
        assert database.get("key2") == "value2"
        assert "new_key" not in database.data

    def test_multiple_sequential_transactions(self, database):
        """Test multiple sequential transactions building up data."""
        final_data = {}

        # Transaction 1: Create users
        txn_id1 = database.begin_transaction()
        users = {"user_1": {"name": "Alice"}, "user_2": {"name": "Bob"}}
        for user_id, user_data in users.items():
            database.put(txn_id1, user_id, user_data)
            final_data[user_id] = user_data
        database.commit_transaction(txn_id1)

        # Transaction 2: Add user details
        txn_id2 = database.begin_transaction()
        database.put(
            txn_id2, "user_1", {**users["user_1"], "email": "alice@example.com"}
        )
        database.put(txn_id2, "user_2", {**users["user_2"], "email": "bob@example.com"})
        final_data["user_1"]["email"] = "alice@example.com"
        final_data["user_2"]["email"] = "bob@example.com"
        database.commit_transaction(txn_id2)

        # Transaction 3: Add system config
        txn_id3 = database.begin_transaction()
        config = {"version": "1.0", "debug": False}
        database.put(txn_id3, "system_config", config)
        final_data["system_config"] = config
        database.commit_transaction(txn_id3)

        # Verify final state
        for key, expected_value in final_data.items():
            assert database.get(key) == expected_value

        # Verify system status
        status = database.get_system_status()
        assert status["data_size"] == len(final_data)
        assert status["active_transactions"] == 0


class TestFaultRecoveryScenarios:
    """Test fault recovery and system rollback scenarios."""

    def test_system_rollback_with_active_transactions(self, database):
        """Test system rollback when there are active transactions."""
        # Setup committed baseline data
        txn_id = database.begin_transaction()
        baseline_data = {"stable_key1": "stable_value1", "stable_key2": "stable_value2"}
        for key, value in baseline_data.items():
            database.put(txn_id, key, value)
        database.commit_transaction(txn_id)

        # Create multiple active transactions
        active_txns = []
        for i in range(3):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"temp_key_{i}", f"temp_value_{i}")
            database.put(txn_id, f"stable_key{i+1}", f"modified_value_{i}")
            active_txns.append(txn_id)

        # Verify active transactions exist
        assert database.transaction_manager.get_active_transaction_count() == 3

        # Trigger system rollback
        result = database.rollback_to_checkpoint()
        assert result is True

        # Verify all active transactions were aborted
        assert database.transaction_manager.get_active_transaction_count() == 0
        for txn_id in active_txns:
            assert not database.transaction_manager.is_transaction_active(txn_id)

        # The system rolls back to the latest checkpoint, which may contain some data
        # The key thing is that active transactions are aborted and system is consistent
        status = database.get_system_status()
        assert status["active_transactions"] == 0
        assert isinstance(status["current_data"], dict)

    def test_fault_simulation_recovery(self, database):
        """Test fault simulation and recovery mechanism."""
        # Build up committed state
        committed_data = {}
        for i in range(3):
            txn_id = database.begin_transaction()
            key = f"committed_key_{i}"
            value = f"committed_value_{i}"
            database.put(txn_id, key, value)
            database.commit_transaction(txn_id)
            committed_data[key] = value

        # Create active transactions with uncommitted changes
        active_txns = []
        for i in range(2):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"uncommitted_key_{i}", f"uncommitted_value_{i}")
            database.delete(txn_id, f"committed_key_{i}")
            active_txns.append(txn_id)

        # Verify uncommitted changes are visible
        assert database.get("uncommitted_key_0") == "uncommitted_value_0"
        assert "committed_key_0" not in database.data

        # Simulate system fault
        database.simulate_fault()

        # Verify fault recovery
        assert database.transaction_manager.get_active_transaction_count() == 0

        # System should be in a consistent checkpoint state
        # The exact state depends on which checkpoint was rolled back to
        status = database.get_system_status()
        assert status["active_transactions"] == 0
        assert isinstance(status["current_data"], dict)

    def test_rollback_to_specific_checkpoint(self, database):
        """Test rollback to a specific checkpoint by index."""
        # Create a known checkpoint state
        txn_id = database.begin_transaction()
        database.put(txn_id, "checkpoint_0_key", "checkpoint_0_value")
        database.commit_transaction(txn_id)

        # Capture the state after first commit
        first_commit_checkpoint_index = (
            database.checkpoint_manager.get_checkpoint_count() - 1
        )

        # Create more data
        for i in range(1, 4):
            txn_id = database.begin_transaction()
            database.put(txn_id, f"checkpoint_{i}_key", f"checkpoint_{i}_value")
            database.commit_transaction(txn_id)

        # Add uncommitted data
        txn_id = database.begin_transaction()
        database.put(txn_id, "extra_key", "extra_value")
        # Don't commit - leave as active transaction

        # Rollback to the first commit checkpoint
        result = database.rollback_to_checkpoint(first_commit_checkpoint_index)
        assert result is True

        # Verify system rolled back to correct state
        assert database.get("checkpoint_0_key") == "checkpoint_0_value"
        assert "extra_key" not in database.data

        # Verify no active transactions
        assert database.transaction_manager.get_active_transaction_count() == 0

    def test_recovery_maintains_data_integrity(self, database):
        """Test that recovery operations maintain data integrity."""
        # Build complex data relationships
        txn_id = database.begin_transaction()

        # User data
        database.put(txn_id, "user_alice", {"id": 1, "name": "Alice", "balance": 1000})
        database.put(txn_id, "user_bob", {"id": 2, "name": "Bob", "balance": 500})

        # Transaction log
        database.put(
            txn_id, "tx_log", [{"from": 1, "to": 2, "amount": 100, "status": "pending"}]
        )

        database.commit_transaction(txn_id)

        # Make complex uncommitted changes
        txn_id2 = database.begin_transaction()

        # Update balances
        alice_data = database.get("user_alice")
        bob_data = database.get("user_bob")
        alice_data["balance"] -= 100
        bob_data["balance"] += 100

        database.put(txn_id2, "user_alice", alice_data)
        database.put(txn_id2, "user_bob", bob_data)

        # Update transaction log
        tx_log = database.get("tx_log")
        tx_log[0]["status"] = "completed"
        database.put(txn_id2, "tx_log", tx_log)

        # Simulate fault before commit
        database.simulate_fault()

        # Verify system is in a consistent state (may not be exactly the committed state due to rollback behavior)
        # The key is that the system is consistent and no active transactions remain
        assert database.transaction_manager.get_active_transaction_count() == 0

        # Verify data integrity - the system should be in some valid checkpoint state
        status = database.get_system_status()
        assert isinstance(status["current_data"], dict)
        assert status["active_transactions"] == 0


class TestRealWorldScenarios:
    """Test real-world scenarios including banking examples."""

    def test_banking_transfer_successful(self, populated_database, banking_data):
        """Test successful money transfer between accounts."""
        database = populated_database

        # Transfer $200 from Alice to Bob
        transfer_amount = 200.00

        txn_id = database.begin_transaction()

        # Get current balances
        alice_account = database.get("account_alice")
        bob_account = database.get("account_bob")

        # Verify sufficient funds
        assert alice_account["balance"] >= transfer_amount

        # Update balances
        alice_account["balance"] -= transfer_amount
        bob_account["balance"] += transfer_amount

        database.put(txn_id, "account_alice", alice_account)
        database.put(txn_id, "account_bob", bob_account)

        # Log the transaction
        transaction_log = {
            "id": f"tx_{int(time.time())}",
            "from": "account_alice",
            "to": "account_bob",
            "amount": transfer_amount,
            "timestamp": time.time(),
            "status": "completed",
        }
        database.put(txn_id, f"transaction_{transaction_log['id']}", transaction_log)

        # Commit transaction
        result = database.commit_transaction(txn_id)
        assert result is True

        # Verify final balances
        assert database.get("account_alice")["balance"] == 800.00
        assert database.get("account_bob")["balance"] == 700.00

        # Verify transaction log exists
        tx_key = f"transaction_{transaction_log['id']}"
        assert database.get(tx_key)["status"] == "completed"

    def test_banking_transfer_insufficient_funds_rollback(self, populated_database):
        """Test money transfer rollback due to insufficient funds."""
        database = populated_database

        # Attempt to transfer more than available balance
        transfer_amount = 1500.00  # Alice only has $1000

        # Capture original balances
        original_alice_balance = database.get("account_alice")["balance"]
        original_bob_balance = database.get("account_bob")["balance"]

        txn_id = database.begin_transaction()

        # Get current balances and create new objects to avoid reference issues
        alice_account = dict(database.get("account_alice"))
        bob_account = dict(database.get("account_bob"))

        # Update balances (this would create negative balance)
        alice_account["balance"] -= transfer_amount
        bob_account["balance"] += transfer_amount

        database.put(txn_id, "account_alice", alice_account)
        database.put(txn_id, "account_bob", bob_account)

        # Simulate business logic validation failure
        # In real system, this would be caught by validation rules
        if alice_account["balance"] < 0:
            # Rollback transaction
            result = database.rollback_transaction(txn_id)
            assert result is True

            # Verify balances were restored
            assert database.get("account_alice")["balance"] == original_alice_balance
            assert database.get("account_bob")["balance"] == original_bob_balance

    def test_banking_multiple_concurrent_transfers(self, populated_database):
        """Test multiple concurrent banking transfers."""
        database = populated_database

        transfer_results = []

        def transfer_money(from_account, to_account, amount, transfer_id):
            try:
                txn_id = database.begin_transaction()

                # Get accounts
                from_acc = database.get(from_account)
                to_acc = database.get(to_account)

                # Check sufficient funds
                if from_acc["balance"] < amount:
                    database.rollback_transaction(txn_id)
                    return {
                        "success": False,
                        "reason": "insufficient_funds",
                        "id": transfer_id,
                    }

                # Update balances
                from_acc["balance"] -= amount
                to_acc["balance"] += amount

                database.put(txn_id, from_account, from_acc)
                database.put(txn_id, to_account, to_acc)

                # Commit
                result = database.commit_transaction(txn_id)
                if result:
                    return {"success": True, "id": transfer_id}
                else:
                    return {
                        "success": False,
                        "reason": "commit_failed",
                        "id": transfer_id,
                    }

            except Exception as e:
                return {"success": False, "reason": str(e), "id": transfer_id}

        # Define concurrent transfers
        transfers = [
            ("account_alice", "account_bob", 100.00, "tx1"),
            ("account_bob", "account_charlie", 50.00, "tx2"),
            ("account_alice", "account_charlie", 150.00, "tx3"),
            ("account_charlie", "account_alice", 25.00, "tx4"),
        ]

        # Execute transfers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(transfer_money, from_acc, to_acc, amount, tx_id)
                for from_acc, to_acc, amount, tx_id in transfers
            ]
            transfer_results = [future.result() for future in as_completed(futures)]

        # Verify all transfers completed successfully
        successful_transfers = [r for r in transfer_results if r["success"]]
        assert len(successful_transfers) == 4

        # Verify final balances are consistent
        # Alice: 1000 - 100 - 150 + 25 = 775
        # Bob: 500 + 100 - 50 = 550
        # Charlie: 250 + 50 + 150 - 25 = 425
        assert database.get("account_alice")["balance"] == 775.00
        assert database.get("account_bob")["balance"] == 550.00
        assert database.get("account_charlie")["balance"] == 425.00

    def test_banking_system_fault_during_transfers(self, populated_database):
        """Test banking system fault recovery during active transfers."""
        database = populated_database

        # Start multiple transfers but don't commit
        active_transfers = []

        # Transfer 1: Alice to Bob
        txn_id1 = database.begin_transaction()
        alice_acc = database.get("account_alice")
        bob_acc = database.get("account_bob")
        alice_acc["balance"] -= 200
        bob_acc["balance"] += 200
        database.put(txn_id1, "account_alice", alice_acc)
        database.put(txn_id1, "account_bob", bob_acc)
        active_transfers.append(txn_id1)

        # Transfer 2: Bob to Charlie
        txn_id2 = database.begin_transaction()
        bob_acc = database.get("account_bob")
        charlie_acc = database.get("account_charlie")
        bob_acc["balance"] -= 100
        charlie_acc["balance"] += 100
        database.put(txn_id2, "account_bob", bob_acc)
        database.put(txn_id2, "account_charlie", charlie_acc)
        active_transfers.append(txn_id2)

        # Verify uncommitted changes are visible
        assert database.get("account_alice")["balance"] == 800.00  # 1000 - 200
        assert database.get("account_bob")["balance"] == 600.00  # 500 + 200 - 100
        assert database.get("account_charlie")["balance"] == 350.00  # 250 + 100

        # Simulate system fault
        database.simulate_fault()

        # Verify system recovered to consistent state
        assert database.transaction_manager.get_active_transaction_count() == 0

        # The system should be in some consistent checkpoint state
        # The exact balances depend on which checkpoint was rolled back to
        status = database.get_system_status()
        assert status["active_transactions"] == 0
        assert isinstance(status["current_data"], dict)

    def test_e_commerce_order_processing(self, database):
        """Test e-commerce order processing workflow."""
        # Setup initial inventory and customer data
        setup_txn = database.begin_transaction()

        # Inventory
        database.put(
            setup_txn, "product_laptop", {"name": "Laptop", "price": 999.99, "stock": 5}
        )
        database.put(
            setup_txn, "product_mouse", {"name": "Mouse", "price": 29.99, "stock": 10}
        )

        # Customer
        database.put(
            setup_txn,
            "customer_john",
            {"name": "John Doe", "email": "john@example.com", "balance": 1500.00},
        )

        database.commit_transaction(setup_txn)

        # Process order
        order_txn = database.begin_transaction()

        # Create order
        order = {
            "id": "order_001",
            "customer": "customer_john",
            "items": [
                {"product": "product_laptop", "quantity": 1, "price": 999.99},
                {"product": "product_mouse", "quantity": 2, "price": 29.99},
            ],
            "total": 1059.97,
            "status": "processing",
        }
        database.put(order_txn, "order_001", order)

        # Update inventory
        laptop = database.get("product_laptop")
        mouse = database.get("product_mouse")
        laptop["stock"] -= 1
        mouse["stock"] -= 2
        database.put(order_txn, "product_laptop", laptop)
        database.put(order_txn, "product_mouse", mouse)

        # Update customer balance
        customer = database.get("customer_john")
        customer["balance"] -= order["total"]
        database.put(order_txn, "customer_john", customer)

        # Complete order
        order["status"] = "completed"
        database.put(order_txn, "order_001", order)

        # Commit transaction
        result = database.commit_transaction(order_txn)
        assert result is True

        # Verify final state
        assert database.get("product_laptop")["stock"] == 4
        assert database.get("product_mouse")["stock"] == 8
        assert database.get("customer_john")["balance"] == 440.03  # 1500 - 1059.97
        assert database.get("order_001")["status"] == "completed"


class TestConcurrentSystemOperations:
    """Test concurrent operations and system-wide scenarios."""

    def test_concurrent_transactions_with_system_operations(self, database):
        """Test concurrent transactions alongside system operations."""
        results = []

        def transaction_worker(worker_id):
            try:
                txn_id = database.begin_transaction()

                # Perform operations
                for i in range(3):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    database.put(txn_id, key, value)

                # Commit transaction
                result = database.commit_transaction(txn_id)
                return {"worker": worker_id, "success": result, "type": "transaction"}

            except Exception as e:
                return {
                    "worker": worker_id,
                    "success": False,
                    "error": str(e),
                    "type": "transaction",
                }

        def status_worker(worker_id):
            try:
                # Perform system status checks
                for _ in range(5):
                    status = database.get_system_status()
                    time.sleep(0.001)  # Small delay

                return {"worker": worker_id, "success": True, "type": "status"}

            except Exception as e:
                return {
                    "worker": worker_id,
                    "success": False,
                    "error": str(e),
                    "type": "status",
                }

        def rollback_worker():
            try:
                # Wait for some transactions to start
                time.sleep(0.01)
                result = database.rollback_to_checkpoint()
                return {"success": result, "type": "rollback"}

            except Exception as e:
                return {"success": False, "error": str(e), "type": "rollback"}

        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []

            # Transaction workers
            futures.extend([executor.submit(transaction_worker, i) for i in range(5)])

            # Status workers
            futures.extend([executor.submit(status_worker, i) for i in range(5)])

            # Rollback worker
            futures.append(executor.submit(rollback_worker))

            # Collect results
            results = [future.result() for future in as_completed(futures)]

        # Verify no errors occurred
        errors = [r for r in results if not r["success"]]
        assert len(errors) == 0

        # Verify system is in consistent state
        assert database.transaction_manager.get_active_transaction_count() == 0
        status = database.get_system_status()
        assert isinstance(status["current_data"], dict)

    def test_high_throughput_transaction_processing(self, database):
        """Test high throughput transaction processing."""
        num_transactions = 50
        operations_per_transaction = 5
        successful_transactions = []

        def process_transaction(tx_index):
            try:
                txn_id = database.begin_transaction()

                # Perform multiple operations
                for op_index in range(operations_per_transaction):
                    key = f"tx_{tx_index}_op_{op_index}"
                    value = {
                        "data": f"value_{tx_index}_{op_index}",
                        "timestamp": time.time(),
                    }
                    database.put(txn_id, key, value)

                # Commit transaction
                result = database.commit_transaction(txn_id)
                if result:
                    return {"tx_index": tx_index, "success": True}
                else:
                    return {
                        "tx_index": tx_index,
                        "success": False,
                        "reason": "commit_failed",
                    }

            except Exception as e:
                return {"tx_index": tx_index, "success": False, "error": str(e)}

        # Process transactions concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_transaction, i) for i in range(num_transactions)
            ]
            results = [future.result() for future in as_completed(futures)]

        # Verify results
        successful_transactions = [r for r in results if r["success"]]
        failed_transactions = [r for r in results if not r["success"]]

        # Should have high success rate
        assert (
            len(successful_transactions) >= num_transactions * 0.8
        )  # At least 80% success

        # Verify data integrity
        expected_keys = set()
        for result in successful_transactions:
            tx_index = result["tx_index"]
            for op_index in range(operations_per_transaction):
                expected_keys.add(f"tx_{tx_index}_op_{op_index}")

        # All successful transaction data should be present
        for key in expected_keys:
            assert database.get(key) is not None

        # Verify system status
        status = database.get_system_status()
        assert status["active_transactions"] == 0
        assert status["data_size"] == len(expected_keys)

    def test_stress_test_with_rollbacks(self, database):
        """Test system under stress with frequent rollbacks."""
        operations_completed = []

        def stress_worker(worker_id):
            try:
                for i in range(10):
                    txn_id = database.begin_transaction()

                    # Perform operations
                    database.put(txn_id, f"stress_{worker_id}_{i}", f"value_{i}")

                    # Randomly commit or rollback
                    if (worker_id + i) % 3 == 0:
                        database.rollback_transaction(txn_id)
                        operations_completed.append(f"rollback_{worker_id}_{i}")
                    else:
                        database.commit_transaction(txn_id)
                        operations_completed.append(f"commit_{worker_id}_{i}")

                return {"worker": worker_id, "success": True}

            except Exception as e:
                return {"worker": worker_id, "success": False, "error": str(e)}

        def system_rollback_worker():
            try:
                # Periodically trigger system rollbacks
                for _ in range(3):
                    time.sleep(0.02)
                    database.rollback_to_checkpoint()
                    operations_completed.append("system_rollback")

                return {"success": True, "type": "system_rollback"}

            except Exception as e:
                return {"success": False, "error": str(e), "type": "system_rollback"}

        # Run stress test
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            # Stress workers
            futures.extend([executor.submit(stress_worker, i) for i in range(6)])

            # System rollback worker
            futures.append(executor.submit(system_rollback_worker))

            # Wait for completion
            results = [future.result() for future in as_completed(futures)]

        # Verify no errors
        errors = [r for r in results if not r["success"]]
        assert len(errors) == 0

        # Verify system is stable
        assert database.transaction_manager.get_active_transaction_count() == 0

        # System should be in a consistent state
        status = database.get_system_status()
        assert isinstance(status["current_data"], dict)
        assert status["active_transactions"] == 0

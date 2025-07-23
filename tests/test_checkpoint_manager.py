"""Unit tests for CheckpointManager."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from src.checkpoint_manager import CheckpointManager
from src.models.checkpoint import Checkpoint


class TestCheckpointManagerSetup:
    """Test CheckpointManager initialization and basic setup."""

    def test_init_default_max_checkpoints(self):
        """Test CheckpointManager initialization with default max_checkpoints."""
        manager = CheckpointManager()
        assert manager.max_checkpoints == 10
        assert manager.checkpoints == []
        assert manager.checkpoint_counter == 0
        assert manager.lock is not None

    def test_init_custom_max_checkpoints(self):
        """Test CheckpointManager initialization with custom max_checkpoints."""
        manager = CheckpointManager(max_checkpoints=5)
        assert manager.max_checkpoints == 5
        assert manager.checkpoints == []
        assert manager.checkpoint_counter == 0


class TestCheckpointCreation:
    """Test checkpoint creation functionality."""

    @pytest.fixture
    def checkpoint_manager(self):
        """Provide a fresh CheckpointManager instance for testing."""
        return CheckpointManager(max_checkpoints=5)

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for checkpoint testing."""
        return {
            "user_1": {"name": "Alice", "balance": 1000},
            "user_2": {"name": "Bob", "balance": 500},
        }

    def test_create_checkpoint_basic(self, checkpoint_manager, sample_data):
        """Test basic checkpoint creation."""
        description = "Test checkpoint"

        checkpoint = checkpoint_manager.create_checkpoint(sample_data, description)

        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.description == description
        assert checkpoint.data == sample_data
        assert checkpoint.data is not sample_data  # Should be deep copy
        assert checkpoint.timestamp > 0
        assert checkpoint.transaction_id.startswith("checkpoint_1_")
        assert len(checkpoint_manager.checkpoints) == 1

    def test_create_checkpoint_without_description(
        self, checkpoint_manager, sample_data
    ):
        """Test checkpoint creation without description."""
        checkpoint = checkpoint_manager.create_checkpoint(sample_data)

        assert checkpoint.description == ""
        assert checkpoint.data == sample_data

    def test_create_checkpoint_data_isolation(self, checkpoint_manager):
        """Test that checkpoint data is isolated from original data."""
        original_data = {"key": {"nested": "value"}}

        checkpoint = checkpoint_manager.create_checkpoint(original_data)

        # Modify original data
        original_data["key"]["nested"] = "modified"

        # Checkpoint data should remain unchanged
        assert checkpoint.data["key"]["nested"] == "value"

    def test_create_multiple_checkpoints(self, checkpoint_manager, sample_data):
        """Test creating multiple checkpoints."""
        descriptions = ["First checkpoint", "Second checkpoint", "Third checkpoint"]

        for i, desc in enumerate(descriptions):
            checkpoint = checkpoint_manager.create_checkpoint(sample_data, desc)
            assert checkpoint.description == desc
            assert len(checkpoint_manager.checkpoints) == i + 1

    @patch("time.time")
    def test_checkpoint_id_generation(self, mock_time, checkpoint_manager, sample_data):
        """Test checkpoint ID generation with mocked time."""
        mock_time.return_value = 1234567890.123

        checkpoint1 = checkpoint_manager.create_checkpoint(sample_data, "First")
        checkpoint2 = checkpoint_manager.create_checkpoint(sample_data, "Second")

        assert checkpoint1.transaction_id == "checkpoint_1_1234567890123"
        assert checkpoint2.transaction_id == "checkpoint_2_1234567890123"

    @patch("builtins.print")
    def test_checkpoint_creation_logging(
        self, mock_print, checkpoint_manager, sample_data
    ):
        """Test that checkpoint creation logs appropriately."""
        description = "Test checkpoint"

        checkpoint = checkpoint_manager.create_checkpoint(sample_data, description)

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Checkpoint created: Test checkpoint at" in call_args
        assert str(checkpoint.timestamp) in call_args


class TestCheckpointLimits:
    """Test checkpoint limit enforcement."""

    @pytest.fixture
    def limited_manager(self):
        """Provide a CheckpointManager with small limit for testing."""
        return CheckpointManager(max_checkpoints=3)

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for checkpoint testing."""
        return {"test": "data"}

    def test_checkpoint_limit_enforcement(self, limited_manager, sample_data):
        """Test that checkpoint limit is enforced."""
        # Create checkpoints up to limit
        for i in range(3):
            limited_manager.create_checkpoint(sample_data, f"Checkpoint {i}")

        assert len(limited_manager.checkpoints) == 3

        # Create one more checkpoint - should remove oldest
        limited_manager.create_checkpoint(sample_data, "Checkpoint 3")

        assert len(limited_manager.checkpoints) == 3
        assert limited_manager.checkpoints[0].description == "Checkpoint 1"
        assert limited_manager.checkpoints[-1].description == "Checkpoint 3"

    def test_checkpoint_limit_multiple_removals(self, limited_manager, sample_data):
        """Test checkpoint limit with multiple excess checkpoints."""
        # Create many checkpoints
        for i in range(10):
            limited_manager.create_checkpoint(sample_data, f"Checkpoint {i}")

        assert len(limited_manager.checkpoints) == 3
        # Should keep the last 3 checkpoints
        assert limited_manager.checkpoints[0].description == "Checkpoint 7"
        assert limited_manager.checkpoints[1].description == "Checkpoint 8"
        assert limited_manager.checkpoints[2].description == "Checkpoint 9"


class TestCheckpointRetrieval:
    """Test checkpoint retrieval functionality."""

    @pytest.fixture
    def populated_manager(self):
        """Provide a CheckpointManager with sample checkpoints."""
        manager = CheckpointManager(max_checkpoints=5)
        sample_data = {"test": "data"}

        for i in range(3):
            manager.create_checkpoint(sample_data, f"Checkpoint {i}")

        return manager

    def test_get_latest_checkpoint_default(self, populated_manager):
        """Test getting latest checkpoint with default index."""
        checkpoint = populated_manager.get_checkpoint()
        assert checkpoint.description == "Checkpoint 2"

    def test_get_checkpoint_by_positive_index(self, populated_manager):
        """Test getting checkpoint by positive index."""
        checkpoint = populated_manager.get_checkpoint(0)
        assert checkpoint.description == "Checkpoint 0"

        checkpoint = populated_manager.get_checkpoint(1)
        assert checkpoint.description == "Checkpoint 1"

    def test_get_checkpoint_by_negative_index(self, populated_manager):
        """Test getting checkpoint by negative index."""
        checkpoint = populated_manager.get_checkpoint(-1)
        assert checkpoint.description == "Checkpoint 2"

        checkpoint = populated_manager.get_checkpoint(-2)
        assert checkpoint.description == "Checkpoint 1"

        checkpoint = populated_manager.get_checkpoint(-3)
        assert checkpoint.description == "Checkpoint 0"

    def test_get_checkpoint_empty_list(self):
        """Test getting checkpoint when no checkpoints exist."""
        manager = CheckpointManager()

        with pytest.raises(ValueError, match="No checkpoints available"):
            manager.get_checkpoint()

    def test_get_checkpoint_invalid_positive_index(self, populated_manager):
        """Test getting checkpoint with invalid positive index."""
        with pytest.raises(ValueError, match="Invalid checkpoint index: 5"):
            populated_manager.get_checkpoint(5)

    def test_get_checkpoint_invalid_negative_index(self, populated_manager):
        """Test getting checkpoint with invalid negative index."""
        with pytest.raises(ValueError, match="Invalid checkpoint index: -1"):
            populated_manager.get_checkpoint(-4)


class TestCheckpointUtilityMethods:
    """Test utility methods for checkpoint information."""

    @pytest.fixture
    def checkpoint_manager(self):
        """Provide a fresh CheckpointManager instance."""
        return CheckpointManager()

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        return {"test": "data"}

    def test_get_checkpoint_count_empty(self, checkpoint_manager):
        """Test checkpoint count when empty."""
        assert checkpoint_manager.get_checkpoint_count() == 0

    def test_get_checkpoint_count_with_checkpoints(
        self, checkpoint_manager, sample_data
    ):
        """Test checkpoint count with checkpoints."""
        for i in range(3):
            checkpoint_manager.create_checkpoint(sample_data, f"Checkpoint {i}")

        assert checkpoint_manager.get_checkpoint_count() == 3

    def test_has_checkpoints_empty(self, checkpoint_manager):
        """Test has_checkpoints when empty."""
        assert checkpoint_manager.has_checkpoints() is False

    def test_has_checkpoints_with_data(self, checkpoint_manager, sample_data):
        """Test has_checkpoints with data."""
        checkpoint_manager.create_checkpoint(sample_data, "Test")
        assert checkpoint_manager.has_checkpoints() is True

    def test_get_latest_checkpoint_description_empty(self, checkpoint_manager):
        """Test getting latest checkpoint description when empty."""
        assert (
            checkpoint_manager.get_latest_checkpoint_description() == "No checkpoints"
        )

    def test_get_latest_checkpoint_description_with_data(
        self, checkpoint_manager, sample_data
    ):
        """Test getting latest checkpoint description with data."""
        checkpoint_manager.create_checkpoint(sample_data, "First checkpoint")
        checkpoint_manager.create_checkpoint(sample_data, "Latest checkpoint")

        assert (
            checkpoint_manager.get_latest_checkpoint_description()
            == "Latest checkpoint"
        )


class TestThreadSafety:
    """Test thread safety of CheckpointManager operations."""

    @pytest.fixture
    def checkpoint_manager(self):
        """Provide a CheckpointManager for thread safety testing."""
        return CheckpointManager(max_checkpoints=100)

    def test_concurrent_checkpoint_creation(self, checkpoint_manager):
        """Test concurrent checkpoint creation from multiple threads."""
        sample_data = {"test": "data"}
        num_threads = 10
        checkpoints_per_thread = 5
        results = []

        def create_checkpoints(thread_id):
            thread_results = []
            for i in range(checkpoints_per_thread):
                checkpoint = checkpoint_manager.create_checkpoint(
                    sample_data, f"Thread {thread_id} Checkpoint {i}"
                )
                thread_results.append(checkpoint)
            results.extend(thread_results)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=create_checkpoints, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all checkpoints were created
        assert len(results) == num_threads * checkpoints_per_thread
        assert (
            checkpoint_manager.get_checkpoint_count()
            == num_threads * checkpoints_per_thread
        )

        # Verify all transaction IDs are unique
        transaction_ids = [cp.transaction_id for cp in results]
        assert len(set(transaction_ids)) == len(transaction_ids)

    def test_concurrent_checkpoint_retrieval(self, checkpoint_manager):
        """Test concurrent checkpoint retrieval from multiple threads."""
        sample_data = {"test": "data"}

        # Create some checkpoints first
        for i in range(10):
            checkpoint_manager.create_checkpoint(sample_data, f"Checkpoint {i}")

        results = []
        errors = []

        def retrieve_checkpoints():
            try:
                # Try various retrieval operations
                latest = checkpoint_manager.get_checkpoint()
                first = checkpoint_manager.get_checkpoint(0)
                count = checkpoint_manager.get_checkpoint_count()
                has_checkpoints = checkpoint_manager.has_checkpoints()
                description = checkpoint_manager.get_latest_checkpoint_description()

                results.append(
                    {
                        "latest": latest,
                        "first": first,
                        "count": count,
                        "has_checkpoints": has_checkpoints,
                        "description": description,
                    }
                )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=retrieve_checkpoints)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 5

        # All results should be consistent
        for result in results:
            assert result["count"] == 10
            assert result["has_checkpoints"] is True
            assert result["description"] == "Checkpoint 9"
            assert result["latest"].description == "Checkpoint 9"
            assert result["first"].description == "Checkpoint 0"

    def test_concurrent_mixed_operations(self, checkpoint_manager):
        """Test concurrent mixed create and retrieve operations."""
        sample_data = {"test": "data"}
        operations_completed = []
        errors = []

        def create_operation(thread_id):
            try:
                for i in range(3):
                    checkpoint_manager.create_checkpoint(
                        sample_data, f"Thread {thread_id} CP {i}"
                    )
                operations_completed.append(f"create_{thread_id}")
            except Exception as e:
                errors.append(e)

        def retrieve_operation(thread_id):
            try:
                time.sleep(0.01)  # Small delay to ensure some checkpoints exist
                if checkpoint_manager.has_checkpoints():
                    checkpoint_manager.get_checkpoint()
                    checkpoint_manager.get_checkpoint_count()
                operations_completed.append(f"retrieve_{thread_id}")
            except Exception as e:
                errors.append(e)

        threads = []

        # Start create threads
        for i in range(3):
            thread = threading.Thread(target=create_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Start retrieve threads
        for i in range(3):
            thread = threading.Thread(target=retrieve_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(operations_completed) == 6
        assert checkpoint_manager.get_checkpoint_count() == 9


class TestErrorHandling:
    """Test error handling in CheckpointManager."""

    @pytest.fixture
    def checkpoint_manager(self):
        """Provide a CheckpointManager for error testing."""
        return CheckpointManager()

    def test_create_checkpoint_with_none_data(self, checkpoint_manager):
        """Test creating checkpoint with None data."""
        # Should not raise an error, but create checkpoint with None data
        checkpoint = checkpoint_manager.create_checkpoint(None, "None data test")
        assert checkpoint.data is None

    def test_create_checkpoint_with_empty_data(self, checkpoint_manager):
        """Test creating checkpoint with empty data."""
        checkpoint = checkpoint_manager.create_checkpoint({}, "Empty data test")
        assert checkpoint.data == {}

    def test_get_checkpoint_boundary_conditions(self):
        """Test get_checkpoint with various boundary conditions."""
        manager = CheckpointManager()

        # Test with single checkpoint
        manager.create_checkpoint({"test": "data"}, "Single checkpoint")

        # Valid indices
        assert manager.get_checkpoint(0).description == "Single checkpoint"
        assert manager.get_checkpoint(-1).description == "Single checkpoint"

        # Invalid indices
        with pytest.raises(ValueError, match="Invalid checkpoint index: 1"):
            manager.get_checkpoint(1)

        with pytest.raises(ValueError, match="Invalid checkpoint index: -1"):
            manager.get_checkpoint(-2)

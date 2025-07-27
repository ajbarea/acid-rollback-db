from typing import Callable, Any
from .exceptions.chaos_exception import ChaosException
from .chaos_config import ChaosConfig
from src.database import AcidRollbackDB

class ChaosProxy:

    """Proxy class to inject chaos into database operations.
    This class wraps the AcidRollbackDB and applies chaos
    configurations such as random failures and delays.
    """

    """Initializes the ChaosProxy with a database instance and chaos configuration.
    Args:
        db (AcidRollbackDB): The database instance to wrap.
        config (ChaosConfig): The configuration for chaos operations.
    """
    def __init__(self, db: AcidRollbackDB, config: ChaosConfig):
        self.db = db
        self.chaos = config
    
    def _with_chaos(self, operation: Callable, *args, context: str, **kwargs) -> Any:
        self.chaos.maybe_fail(context)
        self.chaos.maybe_delay(context)
        return operation(*args, **kwargs)
    
    def begin_transaction(self) -> str:
        """Begin a new transaction with chaos injection."""
        return self._with_chaos(self.db.begin_transaction, context="begin_transaction")
    
    def put(self, txn_id: str, key: str, value: Any) -> bool:
        """Put operation with chaos injection."""
        return self._with_chaos(
            self.db.put, txn_id, key, value, context="put"
        )
    
    def get(self, key: str) -> Any:
        """Get operation with chaos injection."""
        return self._with_chaos(self.db.get, key, context="get")
    
    def commit(self, txn_id: str) -> bool:
        """Commit a transaction with chaos injection."""
        return self._with_chaos(self.db.commit_transaction, txn_id, context="commit")
    
    def rollback(self, txn_id: str) -> bool:
        """Rollback a transaction with chaos injection."""
        return self._with_chaos(self.db.rollback_transaction, txn_id, context="rollback")
    
    def delete(self, txn_id: str, key: str) -> bool:
        """Delete operation with chaos injection."""
        return self._with_chaos(
            self.db.delete, txn_id, key, context="delete"
        )
    
    def commit_transaction(self, txn_id: str) -> bool:
        """Commit transaction with chaos injection."""
        return self._with_chaos(self.db.commit_transaction, txn_id, context="commit_transaction")
    
    def print_status(self):
        return self.db.print_status()
    
    def print_chaos_metrics(self):
        return self.chaos.print_metrics()
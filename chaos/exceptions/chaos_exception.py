class ChaosException(Exception):
    """Base class for all exceptions raised by the Chaos library."""
    
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"ChaosException: {self.message}"
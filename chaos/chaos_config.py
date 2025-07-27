import random
import time
from collections import defaultdict
from .exceptions.chaos_exception import ChaosException

class ChaosConfig:

    """
    Configuration class for the Chaos library.
    This class allows you to set parameters for enabling chaos, such as failure rates,
    delay chances, and maximum delays.
    It also collects runtime metrics for observability.
    """

    """Initializes the ChaosConfig with default or specified parameters.
    Args:
        enable_chaos (bool): Whether to enable chaos. Defaults to False.
        failure_rate (float): Probability of a failure occurring. Defaults to 0.1.
        delay_chance (float): Probability of a delay occurring. Defaults to 0.2.
        max_delay (float): Maximum delay time in seconds. Defaults to 2.0.
    """
    def __init__(
            self,
            enabled: bool = False,
            failure_rate: float = 0.1,
            delay_chance: float = 0.2,
            max_delay: float = 2.0
    ):
        self.enabled = enabled
        self.failure_rate = failure_rate
        self.delay_chance = delay_chance
        self.max_delay = max_delay

        # Chaos metrics
        self.total_operations = 0
        self.failures_injected = 0
        self.delays_injected = 0
        self.total_delay_time = 0.0
        self.failures_by_context = defaultdict(int)
        self.delays_by_context = defaultdict(int)

    """
    Randomly injects a failure if chaos is enabled and the random chance meets the failure rate.
    Args:
        context (str): The context in which the failure is being injected, for logging purposes.
    Raises ChaosException: If a failure is injected.
    """
    def maybe_fail(self, context):
        # Randomly raises a ChaosException based on the failure rate.
        self.total_operations += 1
        if self.enabled and random.random() < self.failure_rate:
            self.failures_injected += 1
            self.failures_by_context[context] += 1
            print(f"[CHAOS] Injected failure in {context}")
            raise ChaosException(f"Chaos failure occurred during {context}.")

    """
    Randomly introduces a delay if chaos is enabled and the random chance meets the delay chance.
    This is done by sleeping for a random duration up to the maximum delay.
    Args:
        context (str): The context in which the delay is being injected, for logging purposes.
    """
    def maybe_delay(self, context):
        # Randomly introduces a delay based on the delay chance and maximum delay.
        if self.enabled and random.random() < self.delay_chance:
            delay = random.uniform(0, self.max_delay)
            self.delays_injected += 1
            self.delays_by_context[context] += 1
            self.total_delay_time += delay
            print(f"[CHAOS] Injected delay of {delay:.2f} seconds in {context}")
            time.sleep(delay)

    """
    Returns collected metrics for chaos execution, such as total operations,
    failures injected, delays introduced, and per-context statistics.
    """
    def get_metrics(self):
            return {
                "Summary": {
                    "total_operations": self.total_operations,
                    "failures_injected": self.failures_injected,
                    "delays_injected": self.delays_injected,
                    "total_delay_time": round(self.total_delay_time, 2),
                },
                "Failures by Context": dict(self.failures_by_context),
                "Delays by Context": dict(self.delays_by_context),
            }

    def print_metrics(self):
        metrics = self.get_metrics()
        
        print("\n=== Chaos Metrics Summary ===")
        for key, value in metrics["Summary"].items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
        
        print("\n--- Failures by Context ---")
        if metrics["Failures by Context"]:
            for context, count in metrics["Failures by Context"].items():
                print(f"{context}: {count}")
        else:
            print("No failures recorded.")
        
        print("\n--- Delays by Context ---")
        if metrics["Delays by Context"]:
            for context, count in metrics["Delays by Context"].items():
                print(f"{context}: {count}")
        else:
            print("No delays recorded.")
        print("===========================\n")

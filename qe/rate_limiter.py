import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """
    TokenBucketRateLimiter implements a token bucket algorithm for rate limiting.

    This class allows controlling the rate of actions (such as API calls) by maintaining a bucket of tokens.
    Tokens are added to the bucket at a fixed rate (rate_per_minute), up to a maximum capacity.
    Each action consumes a specified number of tokens; if insufficient tokens are available, the caller waits
    until enough tokens are refilled.

    Thread safety is ensured via a lock, and the class supports pickling by omitting the lock during serialization.
    """

    def __init__(self, rate_per_minute: int, capacity: Optional[int] = None):
        """
        rate_per_minute: Tokens added per minute
        capacity: Max number of tokens in the bucket
        """
        if capacity is None:
            capacity = rate_per_minute
        self.rate = rate_per_minute / 60.0  # Convert rate to tokens per second
        self.capacity = capacity
        self.tokens = 0.0  # Use float to track fractional tokens
        self.lock = threading.Lock()
        self.last_refill = time.perf_counter()

    def acquire(self, tokens: int = 1):
        with self.lock:
            self._refill()
            while self.tokens < tokens:
                # Not enough tokens, wait until next refill
                sleep_time = (tokens - self.tokens) / self.rate
                time.sleep(sleep_time)
                self._refill()
            self.tokens -= tokens

    def _refill(self):
        now = time.perf_counter()
        elapsed = now - self.last_refill
        # Add tokens based on the elapsed time - use float to avoid truncation
        new_tokens = elapsed * self.rate
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    # Implement __getstate__ and __setstate__ to make the class pickleable
    def __getstate__(self):
        # Return the instance state without the lock
        state = self.__dict__.copy()
        # Remove the lock before pickling
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        # Restore the instance state from the unpickled dictionary
        self.__dict__.update(state)
        # Recreate a new lock since it's not pickleable
        self.lock = threading.Lock()
        self.tokens = 0.0

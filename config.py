from enum import Enum

class SamplingStrategy(Enum):
    """Sampling strategies for follower actions."""
    NONE = 0
    ARGMAX = 1
    SOFTMAX = 2
    BOLTZMANN_MULTIPLICATION = 3

class EnsemblingStrategy(Enum):
    NONE = 0
    MAJORITY_VOTING_RAW = 1
    MAJORITY_VOTING_SOFTMAX = 2
    BOLTZMANN_MULTIPLICATION = 3
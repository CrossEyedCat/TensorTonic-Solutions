import numpy as np
def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    values = np.array(values, dtype=float)
    transitions = np.array(transitions, dtype=float)
    rewards = np.array(rewards, dtype=float)
    return np.max(rewards +  gamma * np.sum(transitions*values, axis = 2), axis = 1).tolist()
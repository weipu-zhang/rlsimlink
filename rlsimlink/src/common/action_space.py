"""
Action space classes for handling discrete, continuous, and mixed action spaces.
"""

from typing import Dict, Any, List, Union
import numpy as np


class DiscreteActionSpace:
    """Single dimension discrete action space."""

    def __init__(self, n: int, expand_dim: bool = False):
        """Initialize discrete action space.

        Args:
            n: Number of discrete actions
            expand_dim: Whether to expand dimension when sampling (return array instead of scalar)
        """
        self.dim = 1
        self.n = n
        self.expand_dim = expand_dim

    def sample(self) -> Union[int, np.ndarray]:
        """Sample a random action.

        Returns:
            Random action (int or array depending on expand_dim)
        """
        action = np.random.randint(self.n)
        if self.expand_dim:
            action = np.array([action])
        return action


class ContinuousActionSpace:
    """Multi-dimensional continuous action space."""

    def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple):
        """Initialize continuous action space.

        Args:
            low: Lower bounds for each dimension
            high: Upper bounds for each dimension
            shape: Shape of the action space
        """
        self.dim = shape[0] if len(shape) > 0 else 1
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.shape = shape

    def sample(self) -> np.ndarray:
        """Sample a random action.

        Returns:
            Random action array
        """
        action = np.random.uniform(self.low, self.high, self.dim).astype(np.float32)
        return action


class MixedActionSpace:
    """Mixed discrete and continuous action space."""

    def __init__(self, spaces: List[Dict[str, Any]]):
        """Initialize mixed action space.

        Args:
            spaces: List of space dictionaries, each containing type and parameters
        """
        self.spaces = []
        self.dim = len(spaces)

        for space_info in spaces:
            space_type = space_info.get("type")
            if space_type == "discrete":
                n = space_info.get("n")
                self.spaces.append(DiscreteActionSpace(n, expand_dim=False))
            elif space_type == "continuous":
                low = space_info.get("low")
                high = space_info.get("high")
                shape = space_info.get("shape", (1,))
                self.spaces.append(ContinuousActionSpace(low, high, shape))
            else:
                raise ValueError(f"Unknown action space type: {space_type}")

    def sample(self) -> List[Union[int, np.ndarray]]:
        """Sample a random action from each sub-space.

        Returns:
            List of actions, one for each dimension
        """
        return [space.sample() for space in self.spaces]


class ActionSpace:
    """
    Universal action space that can handle discrete, continuous, or mixed action spaces.

    This class parses the action space dictionary from the server and creates
    appropriate sub-spaces for sampling.
    """

    def __init__(self, action_space_info: Dict[str, Any], expand_dim: bool = False):
        """Initialize action space from server response.

        Args:
            action_space_info: Dictionary containing action space information:
                {
                    "dimensions": int,
                    "spaces": [
                        {"type": "discrete", "n": int} or
                        {"type": "continuous", "low": float/list, "high": float/list, "shape": tuple}
                    ]
                }
            expand_dim: Whether to expand dimension for single discrete actions
        """
        self.dimensions = action_space_info.get("dimensions", 0)
        spaces_info = action_space_info.get("spaces", [])

        if self.dimensions == 0 or len(spaces_info) == 0:
            raise ValueError("Invalid action space info: no dimensions or spaces defined")

        # Check if all spaces are of the same type
        space_types = [space.get("type") for space in spaces_info]
        all_discrete = all(t == "discrete" for t in space_types)
        all_continuous = all(t == "continuous" for t in space_types)

        # Create appropriate action space based on type
        if all_discrete and self.dimensions == 1:
            # Single discrete action space
            n = spaces_info[0].get("n")
            self._space = DiscreteActionSpace(n, expand_dim=expand_dim)
            self.space_type = "discrete"
        elif all_continuous and self.dimensions == 1:
            # Single continuous action space
            low = spaces_info[0].get("low")
            high = spaces_info[0].get("high")
            shape = spaces_info[0].get("shape", (1,))
            self._space = ContinuousActionSpace(low, high, shape)
            self.space_type = "continuous"
        else:
            # Mixed or multi-dimensional action space
            self._space = MixedActionSpace(spaces_info)
            self.space_type = "mixed"

    def sample(self) -> Union[int, np.ndarray, List[Union[int, np.ndarray]]]:
        """Sample a random action from the action space.

        Returns:
            Random action (type depends on the action space type)
        """
        return self._space.sample()

    @property
    def dim(self) -> int:
        """Get the dimension of the action space.

        Returns:
            Number of dimensions
        """
        return self.dimensions

    @property
    def n(self) -> int:
        """Get the number of actions (for discrete spaces only).

        Returns:
            Number of discrete actions

        Raises:
            AttributeError: If not a discrete action space
        """
        if self.space_type == "discrete":
            return self._space.n
        else:
            raise AttributeError("n is only available for discrete action spaces")

    @property
    def low(self) -> np.ndarray:
        """Get the lower bounds (for continuous spaces only).

        Returns:
            Lower bounds array

        Raises:
            AttributeError: If not a continuous action space
        """
        if self.space_type == "continuous":
            return self._space.low
        else:
            raise AttributeError("low is only available for continuous action spaces")

    @property
    def high(self) -> np.ndarray:
        """Get the upper bounds (for continuous spaces only).

        Returns:
            Upper bounds array

        Raises:
            AttributeError: If not a continuous action space
        """
        if self.space_type == "continuous":
            return self._space.high
        else:
            raise AttributeError("high is only available for continuous action spaces")

    def __repr__(self) -> str:
        """String representation of the action space."""
        if self.space_type == "discrete":
            return f"ActionSpace(type=discrete, n={self._space.n}, dim={self.dimensions})"
        elif self.space_type == "continuous":
            return (
                f"ActionSpace(type=continuous, dim={self.dimensions}, low={self._space.low}, high={self._space.high})"
            )
        else:
            return f"ActionSpace(type=mixed, dim={self.dimensions})"

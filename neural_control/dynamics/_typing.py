"""Shared type aliases and protocols for dynamics modules."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float_]
"""Alias for numpy arrays containing float data."""

StateTensor: TypeAlias = torch.Tensor
"""Alias for batched state tensors shaped ``(batch, state_dim)``."""

ActionTensor: TypeAlias = torch.Tensor
"""Alias for batched action tensors shaped ``(batch, action_dim)``."""

ImageTensor: TypeAlias = torch.Tensor
"""Alias for batched image tensors used by image-based dynamics."""

SequenceBufferTensor: TypeAlias = torch.Tensor
"""Alias for stacked state-action histories used by sequence dynamics."""


@runtime_checkable
class DiscreteTimeDynamics(Protocol):
    """Protocol for discrete-time dynamics supporting batched simulation."""

    def __call__(self, state: StateTensor, action: ActionTensor, dt: float) -> StateTensor:
        """Return the next batched state for the given action and step size."""


__all__ = [
    "ActionTensor",
    "DiscreteTimeDynamics",
    "FloatArray",
    "ImageTensor",
    "SequenceBufferTensor",
    "StateTensor",
]

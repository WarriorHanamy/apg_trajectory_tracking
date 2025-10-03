from __future__ import annotations

from typing import Any, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.floating[Any]]
Float32Array = npt.NDArray[np.float32]
ArrayLike = npt.ArrayLike


class DynamicsState:
    def __init__(self) -> None:
        self._position: FloatArray = np.zeros(3, dtype=float)
        self._attitude: Euler = Euler(0.0, 0.0, 0.0)
        self._velocity: FloatArray = np.zeros(3, dtype=float)
        self._rotorspeeds: FloatArray = np.zeros(4, dtype=float)
        self._last_velocity: FloatArray = np.zeros(3, dtype=float)
        self._angular_velocity: FloatArray = np.zeros(3, dtype=float)

    def set_position(self, pos: FloatArray) -> None:
        self._position = pos

    @property
    def position(self) -> FloatArray:
        return self._position

    @property
    def attitude(self) -> Euler:
        return self._attitude

    @property
    def velocity(self) -> FloatArray:
        return self._velocity

    @property
    def rotor_speeds(self) -> FloatArray:
        return self._rotorspeeds

    @property
    def last_velocity(self) -> FloatArray:
        return self._last_velocity

    @property
    def angular_velocity(self) -> FloatArray:
        return self._angular_velocity

    @property
    def net_rotor_speed(self) -> float:
        return (
            self._rotorspeeds[0]
            - self._rotorspeeds[1]
            + self._rotorspeeds[2]
            - self._rotorspeeds[3]
        )

    @property
    def formatted(self) -> dict[str, StateComp]:
        return {
            "position:": self._position,
            "attitude:": self._attitude,
            "velocity:": self._velocity,
            "rotorspeeds:": self._rotorspeeds,
            "angular_velocity:": self._angular_velocity,
        }

    @property
    def as_np(self) -> Float32Array:
        """
        Convert state to np array
        """
        return np.array(
            (
                list(self._position)
                + list(self._attitude._euler)
                + list(self._velocity)
                + list(self._angular_velocity)
            ),
            dtype=np.float32,
        )

    def from_np(self, state_array: FloatArray) -> None:
        """
        Convert np array to dynamic state
        """
        # set last velocity
        self._last_velocity = self._velocity.copy()
        # other normal things
        self._position = state_array[:3]
        self._attitude = Euler(*tuple(state_array[3:6]))
        self._velocity = state_array[6:9]
        self._angular_velocity = state_array[9:]


class Euler:
    """
    Defines an Euler angle (roll, pitch, yaw). We
    do try to cache as many intermediate results
    as possible here (e.g. the transformation matrices).

    Therefore, do not change the `_euler` attribute
    except for using the provided setters!
    """

    def __init__(self, roll: float, pitch: float, yaw: float) -> None:
        self._euler: FloatArray = np.array([roll, pitch, yaw], dtype=float)
        self._cache: dict[str, Any] = {}

    @staticmethod
    def from_numpy_array(array: ArrayLike) -> "Euler":
        array_np = np.asarray(array, dtype=float)
        assert array_np.shape == (3,)
        return Euler(float(array_np[0]), float(array_np[1]), float(array_np[2]))

    @staticmethod
    def zero() -> "Euler":
        return Euler(0.0, 0.0, 0.0)

    @property
    def roll(self) -> float:
        return float(self._euler[0])

    @roll.setter
    def roll(self, value: float) -> None:
        self._euler[0] = value
        self._cache = {}

    @property
    def pitch(self) -> float:
        return float(self._euler[1])

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._euler[1] = value
        self._cache = {}

    @property
    def yaw(self) -> float:
        return float(self._euler[2])

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._euler[2] = value
        self._cache = {}

    def rotate(self, amount: ArrayLike) -> None:
        self._euler = self._euler + np.asarray(amount, dtype=float)
        self._cache = {}

    def rotated(self, amount: ArrayLike) -> "Euler":
        amount_np = np.asarray(amount, dtype=float)
        return Euler(
            float(self.roll + amount_np[0]),
            float(self.pitch + amount_np[1]),
            float(self.yaw + amount_np[2]),
        )

    def add_to_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def get_from_cache(self, key: str) -> Any | None:
        return self._cache.get(key)

    def __repr__(self) -> str:
        return "Euler(roll=%g, pitch=%g, yaw=%g)" % (self.roll, self.pitch, self.yaw)


ObsComp: TypeAlias = FloatArray | Euler  # position/attitude/velocity/angular_velocity/image
Action: TypeAlias = FloatArray
StateComp: TypeAlias = FloatArray | Euler  # position/attitude/velocity/angular_velocity/rotorspeeds
UInt8Image: TypeAlias = npt.NDArray[np.uint8]
FloatImage: TypeAlias = npt.NDArray[np.float_]
RenderImage: TypeAlias = UInt8Image
ImageBuffer: TypeAlias = FloatImage
CartPoleAction: TypeAlias = torch.Tensor | npt.NDArray[np.float_] | Sequence[float] | float

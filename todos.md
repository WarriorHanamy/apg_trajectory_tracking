# TODOs

- [x] Replace `gym` dependency with `gymnasium` across dependency lists and import statements.
- [x] Type Hints for all neural_control/environments/*.py
- []  Type Hints for all neural_control/environments/*.py. 如下，要更注重语义，隐藏实际细节
ObsComp: TypeAlias = (
    FloatArray | Euler
)  # position/attitude/velocity/angular_velocity/image
Action: TypeAlias = FloatArray
StateComp: TypeAlias = (
    FloatArray | Euler
)  # position/attitude/velocity/angular_velocity/rotorspeeds

from attr import dataclass
from tensorflow.python import Shape


@dataclass
class NetworkInput:
    input_shape: Shape
    mean: float
    std: float
    number_of_classes: int

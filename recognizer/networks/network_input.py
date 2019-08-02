from attr import dataclass
from tensorflow.python import Shape


@dataclass
class NetworkInput:
    input_shape: Shape
    number_of_classes: int
    mean: float = None
    std: float = None

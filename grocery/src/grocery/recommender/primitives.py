from dataclasses import dataclass
import numpy as np


type Embedding = np.ndarray
type Feature = float | str | Embedding


@dataclass
class Candidate:
    id: int
    features: dict[str, Feature]


__all__ = ["Candidate", "Embedding", "Feature"]

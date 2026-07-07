from abc import ABC, abstractmethod

class Dataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return the sample at the given index."""
        pass

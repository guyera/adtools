from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch import Tensor

"""
Created Monday 11/16/2020 09:28 AM PST

Authors:
  Alexander Guyer

Description:
  Provides an abstract base class for PyTorch anomaly detectors.
"""

"""
Description:
  Abstract base class for PyTorch anomaly detectors
"""
class AnomalyDetector(ABC):
    """
    Description: 
      Fits the anomaly detector to the training data.
    
    Parameters:
      train_data_loader: DataLoader from which to load training data.
      validation_data_loader: Optional DataLoader from which to load
                              validation data.
      kwargs: Can be used for implementation-specific training-time
              arguments.
    """
    @abstractmethod
    def fit(self, train_data_loader: DataLoader,
            validation_data_loader: DataLoader = None,
            **kwargs) -> None:
        return NotImplemented

    """
    Description:
      Computes scalar anomaly scores for all of the inputs in the given
      DataLoader.
    
    Parameters:
      data_loader: DataLoader from which to load data to be scored.
    
    Returns:
      A Tensor of N anomaly scores, where N is the total number of elements
      iterable by the given DataLoader. The ordering of the anomaly scores
      must match the ordering of elements returned by the DataLoader
      iterator.
    """
    @abstractmethod
    def score(self, data_loader: DataLoader) -> Tensor:
        return NotImplemented

    """
    Description:
      Computes scalar anomaly scores for the corresponding inputs; score()
      should be preferred when possible.
    
    Parameters:
      input: An (N, ...) sized Tensor of data points, where ... indicates
      zero or more free dimensions, for which to compute anomaly scores.

    Returns:
      An N-sized Tensor of anomaly scores corresponding to the given
      inputs.
    """
    @abstractmethod
    def __call__(self, input: Tensor) -> Tensor:
        return NotImplemented

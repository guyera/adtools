from abc import ABC, abstractmethod

from torch import Tensor

"""
Created Monday 11/16/2020 10:20 AM PST

Authors:
  Alexander Guyer

Description:
  Provides an abstract base class for PyTorch oracle anomaly
  detectors
"""

"""
Description:
  Abstract base class for PyTorch oracle anomaly detectors
"""
class OracleAnomalyDetector:
    """
    Description:
      Fits the oracle anomaly detector to the data provided.
    
    Parameters:
      training_data: An (N, ...) sized Tensor of data embeddings against
                     which to fit the oracle anomaly detector
      training_data_targets: An N-sized Tensor of binary target
                             classifications (0|1) corresponding to the
                             N input data points, where 0 represents
                             nominal data, and 1 represents anomalous data.
      validation_data: An optional (N, ...) sized Tensor of data embeddings
                       used for model selection, early stopping, etc.
      validation_data_targets: An N-sized tensor of binary targets for the
                               validation data. 0 represents nominal data,
                               and 1 represents anomalous data.
      kwargs: Can be used for implementation-specific training-time
              arguments.
    """
    @abstractmethod
    def fit(self, training_data: Tensor,
            training_data_targets: Tensor,
            validation_data: Tensor = None,
            validation_data_targets: Tensor = None,
            **kwargs) -> None:
        return NotImplemented
    
    """
    Description:
      Classifies the provided tensor of data points as nominals or anomalies
    
    Parameters:
      data: (N, ...) sized tensor of data points to classify

    Returns:
      An N-sized tensor of binary classifications. 0 represents nominal
      data, and 1 represents anomalous data. The ordering of the
      classifications matches the ordering of the provided data points.
    """
    @abstractmethod
    def classify(self, data: Tensor) -> Tensor:
        return NotImplemented

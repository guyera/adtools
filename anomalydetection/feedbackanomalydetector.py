"""
Created Tuesday 11/17/2020 10:44 AM PST

Authors:
  Alexander Guyer

Description:
  Provides a subclass of AnomalyDetector which is capable of receiving feedback
  during test time and updating accordingly.
"""

from abc import abstractmethod

from torch import Tensor

from adtools.anomalydetection.anomalydetector import AnomalyDetector

"""
Description:
  A subclass of AnomalyDetector which is capable of receiving feedback during
  test time and updating accordingly.
"""
class FeedbackAnomalyDetector(AnomalyDetector):
    """
    Description:
      Receives feedback in the form of a single data point during test time
      and updates the model accordingly.
    
    Parameters:
      input: A Tensor representing the feedback data point
      target: A 1-sized Tensor containing the binary target of the input.
              0 represents nominal data, 1 represents anomalous data.
    """
    @abstractmethod
    def receive_feedback(self, input: Tensor, target: Tensor) -> None:
        return NotImplemented

"""
Created Thursday 11/19/2020 09:20 AM PST

Authors:
  Alexander Guyer

Description:
  Provides an Oracle subclass wrapper around the
  sklearn.neighbors.KNeighborsClassifier
"""

from torch import Tensor, from_numpy
from sklearn.neighbors import KNeighborsClassifier

from adtools.evaluation.oracle.oracle import ProbabilisticOracle

"""
Description:
  ProbabilisticOracle subclass wrapper around the
  sklearn.neighbors.KNeighborsClassifier
"""
class KNeighborsOracle(ProbabilisticOracle):
    """
    Description:
      Constructor

    Parameters:
      All parameters are as described in sklearn.neighbors.KNeighborsClassifier
    """
    def __init__(self, n_neighbors = None, weights = 'uniform',
            algorithm = 'auto', leaf_size = 30, p = 2, metric = 'minkowski',
            metric_params = None, n_jobs = None, **kwargs):
        super().__init__()
        self.classifier = KNeighborsClassifier(n_neighbors = n_neighbors,
                weights = weights, algorithm = algorithm, leaf_size = leaf_size,
                p = p, metric = metric, metric_params = metric_params,
                n_jobs = n_jobs, **kwargs)

    """
    See adtools.evaluation.oracle.oracle.Oracle
    """
    def fit(self, training_data: Tensor, training_data_targets: Tensor,
            validation_data: Tensor = None,
            validation_data_targets: Tensor = None, **kwargs) -> None:
        training_data = training_data.data.cpu().numpy()
        training_data_targets = training_data_targets.data.cpu().numpy()
        # KNeighborsOracle doesn't actually use validation data for early
        # stopping, etc, so the validation arguments can be ignored
        self.classifier.fit(training_data, training_data_targets)

    """
    See adtools.evaluation.oracle.oracle.Oracle
    """
    def predict_probabilities(self, data: Tensor) -> Tensor:
        data = data.data.cpu().numpy()
        probabilities = from_numpy(self.classifier.predict(data))
        probabilities = probabilities.to(self.device) \
                if self.device is not None else probabilities
        return probabilities

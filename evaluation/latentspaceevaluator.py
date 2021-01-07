"""
Created Thursday 11/19/2020 09:56 AM PST

Description:
  Provides an evaluator class which makes use of multiple oracles to score a
  latent space
"""

from typing import Iterable, Tuple
from torch import Tensor

from adtools.evaluation.oracle import Oracle

"""
Description:
  Evaluator class which makes use of multiple oracles to score a latent space

Attributes:
  oracles: Iterable of oracles over which the evaluator will iterate when
           scoring a latent space
"""
class LatentSpaceEvaluator:
    """
    Description:
      Constructor

    Parameters:
      oracles: Corresponds to self.oracles
    """
    def __init__(self, oracles: Iterable[Oracle]):
        self.oracles = oracles

    """
    Description:
      Fits all oracles to the data provided.

    Parameters:
      All parameters correspond to those in
      adtools.evaluation.oracle.oracle.Oracle.fit()
    """
    def __fit(self, oracles: Iterable[Oracle], training_data: Tensor,
            training_data_targets: Tensor, validation_data: Tensor = None,
            validation_data_targets: Tensor = None,
            **kwargs) -> None:
        for oracle in oracles:
            oracle.fit(training_data = training_data,
                    training_data_targets = training_data_targets,
                    validation_data = validation_data,
                    validation_data_targets = validation_data_targets,
                    **kwargs)

    """
    Description:
      Fits the underlying oracles to the data provided

    Parameters:
      training_data: An (N, ...)-sized Tensor against which to fit the
                     underlying oracles
      training_data_targets: An N-sized Tensor of target binary classificaionts
                             corresponding to training_data. 0 corresponds to
                             nominal data, 1 corresponds to anomalous data.
      validation_data: An (N, ...)-sized Tensor of validation data used for
                       early stopping, model selection, etc. during oracle
                       training
      validation_data_targets: An N-sized Tensor of target binary
                               classifications corresponding to validation_data.
                               0 corresponds to nominal data, 1 corresponds to
                               anomalous data.
      kwargs: Passed to the oracles' fit() functions
    """
    def fit(self, training_data: Tensor, training_data_targets: Tensor,
            validation_data: Tensor = None,
            validation_data_targets: Tensor = None, **kwargs) -> None:
        self.__fit(self.oracles, training_data, training_data_targets,
                validation_data, validation_data_targets, **kwargs)

    """
    Description:
      Scores the latent space representation of the given data in the range
      [0, 1] by computing the max over the accuracies of all of the provided
      oracles.

    Parameters:
      data: An (N, ...)-sized Tensor of data against which to test the provided
            oracles and evaluate the latent representation
      targets: An N-sized Tensor of target binary classifications associated
               with data. 0 corresponds to nominal data, 1 corresponds to
               anomalous data.
      oracles: The trained oracles against which to evaluate the latent
               representation

    Returns:
      1: A score in the range [0, 1] for the latent representation
    """
    def __score(self, data: Tensor, targets: Tensor, \
            oracles: Iterable[Oracle]) -> float:
        best_accuracy = 0.0
        for oracle in oracles:
            predictions = oracle.classify(data)
            num_correct = torch.sum((predictions == targets).int()).data
            accuracy = float(num_correct) / predictions.shape[0]
            best_accuracy = accuracy if accuracy > best_accuracy \
                    else best_accuracy
        return best_accuracy
    
    """
    Description:
      Fits the underlying oracles to the training data and then scores the
      latent representation using the test data by computing a max over
      the oracles' classification accuracies

    Parameters:
      testing_data: An (N, ...)-sized Tensor against which to test the
                    underlying oracles and compute a latent representation score
      testing_data_targets: An N-sized Tensor of target binary classifications
                            associated with testing_data. 0 corresponds to
                            nominal data, 1 corresponds to anomalous data.
      training_data: An (N, ...)-sized Tensor against which to fit the
                     underlying oracles. If None, the training stage is skipped.
      training_data_targets: An N-sized Tensor of target binary classifications
                             associated with training_data. 0 corresponds to
                             nominal data, 1 corresponds to anomalous data. If
                             None, the training stage is skipped.
      validation_data: An (N, ...)-sized Tensor used for model selection,
                       early stopping, etc. during oracle training.
      validation_data_targets: An N-sized Tensor of target binary
                               classifications associated with validation_data.
                               0 corresponds to nominal data, 1 corresponds to
                               anomalous data.
      kwargs: Passed to oracles' fit() functions during training
    """
    def score(self, testing_data: Tensor, testing_data_targets: Tensor, 
            training_data: Tensor = None, training_data_targets: Tensor = None,
            validation_data: Tensor = None,
            validation_data_targets: Tensor = None, **kwargs) -> float:
        # Fit the underlying oracles to the data, if desired
        if training_data is not None and training_data_targets is not None:
            self.fit(training_data, training_data_targets, validation_data,
                    validation_data_targets, **kwargs)
        
        # Test the oracles against the testing data to compute a latent
        # representation score
        return self.__score(testing_data, testing_data_targets, self.oracles)

    """
    Description:
      Scores the latent space representation of the given data in the range
      [0, 1] by computing the max over the accuracies of all underlying
      oracles for each bootstrap sample. Deep copies are constructed from each
      oracle for each bootstrap sample of the provided latent representations,
      and the deep copies are fit to the bootstrap data.

    Parameters:
      training_data: An (N, ...)-sized Tensor from which to construct bootstrap
                     samples and fit the oracles
      training_data_targets: An N-sized Tensor of target binary classification
                             targets associated with training_data. 0
                             corresponds to nominal data, 1 corresponds to
                             anomalous data
      testing_data: An (N, ...)-sized Tensor from which to construct bootstrap
                    samples to test the oracles and evaluate the latent
                    representation
      testing_data_targets: An N-sized Tensor of target binary classifications
                            associated with testing_data. 0 corresponds to
                            nominal data, 1 corresponds to anomalous data.
      validation_data: An (N, ...)-sized Tensor from which to construct
                       bootstrap samples to perform early stopping,
                       model selection, etc. during oracle training
      validation_data_targets: An N-sized Tensor of target binary
                               classifications associated with validation_data.
                               0 corresponds to nominal data, 1 corresponds to
                               anomalous data.
      n_bootstrap_samples: Number of bootstrap samples to use during scoring
      bootstrap_sample_size: Size of each bootstrap sample. If None, the size
                             is implicitly set to the number of data points.
      ci_confidence: The confidence with which to construct the score
                     confidence interval
      kwargs: Passed to the oracles when fitting them to the bootstrap training
              samples
    
    Returns:
      1: The mean across bootstrap sample scores
      2: The standard error of the mean across bootstrap sample scores
      3: The confidence interval of bootstrap sample scores
    """
    def score_bootstrap(self, training_data: Tensor,
            training_data_targets: Tensor, testing_data: Tensor,
            testing_data_targets: Tensor, validation_data: Tensor = None,
            validation_data_targets: Tensor = None, n_bootstrap_samples = 10,
            bootstrap_sample_size = None, ci_confidence: float = 0.95,
            **kwargs) -> Tuple[float, float, Tuple[float, float]]:
        scores = np.zeros(shape = n_bootstrap_samples)
        all_training_indices = np.arange(training_data.shape[0])
        all_testing_indices = np.arange(training_data.shape[0])
        all_validation_indices = np.arange(validation_data.shape[0]) \
                if validation_data is not None else None

        for bootstrap_index in range(n_bootstrap_samples):
            # Construct deep copies of each oracle
            oracles = [copy.deepcopy(oracle) for oracle in self.oracles]
            
            # Bootstrap samples of training, testing, and validation data
            bootstrap_training_indices = np.random.choice(all_training_indices,
                    size = bootstrap_sample_size)
            bootstrap_training_data = training_data[bootstrap_training_indices]
            bootstrap_training_data_targets = \
                    training_data_targets[bootstrap_training_indices]
            
            bootstrap_testing_indices = np.random.choice(all_testing_indices,
                    size = bootstrap_sample_size)
            bootstrap_testing_data = testing_data[bootstrap_testing_indices]
            bootstrap_testing_data_targets = \
                    testing_data_targets[bootstrap_testing_indices]

            if validation_data is not None and \
                        validation_data_targets is not None:
                bootstrap_validation_indices = np.random.choice(
                        all_validation_indices, size = bootstrap_sample_size)
                bootstrap_validation_data = \
                        validation_data[bootstrap_validation_indices]
                bootstrap_validation_data_targets = \
                        validation_data_targets[bootstrap_validation_indices]
            else:
                bootstrap_validation_data = None
                bootstrap_validation_data_targets = None
            
            # Fit the oracles to the bootstrap training and validation samples
            self.__fit(oracles, training_data, training_data_targets,
                    validation_data, validation_data_targets, **kwargs)
            
            # Test the oracles against the bootstrap testing sample to compute
            # a latent representation score
            scores[bootstrap_idx] = self.__score(testing_data,
                    testing_data_targets, oracles)

        # Construct confidence intervals assuming the scores are normally
        # distributed
        mean = np.mean(scores)
        standard_error = scipy.stats.sem(scores)
        percentile_distance = standard_error * scipy.stats.t.ppf(
                (1 + ci_confidence) / 2,
                n_bootstrap_samples - 1)
        
        return mean, standard_error, \
                (mean - percentile_distance, mean + percentile_distance)

"""
Created Thursday 11/19/2020 09:20 AM PST

Authors:
  Alexander Guyer

Description:
  Provides an Oracle subclass wrapper around the
  sklearn.neighbors.KNeighborsClassifier
"""

import torch

from adtools.evaluation.oracle.oracle import ProbabilisticOracle

"""
Description:
  Multilayer perceptron PyTorch implementation
"""
class _MultilayerPerceptron(torch.nn.Module):
    """
    Description:
      Constructor
    
    Parameters:
      input_size: Dimension of flattened input space
      n_classes: Number of classes
      hidden_layers: List of hidden layer sizes. [128] means a single hidden
                     layer with 128 nodes.
    """
    def __init__(self, input_size, n_classes, hidden_layers = [128]):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_layers[0])
        self.input_sigmoid = torch.nn.Sigmoid()
        self.hidden_layers = torch.nn.ModuleList([])
        for idx in range(0, len(hidden_layers) - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers[idx],
                    hidden_layers[idx + 1]))
            self.hidden_layers.append(torch.nn.Sigmoid())
        self.output_layer = torch.nn.Linear(hidden_layers[-1], n_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_sigmoid(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x

"""
Description:
  ProbabilisticOracle subclass; pytorch multilayer perceptron
"""
class MultilayerPerceptronOracle(ProbabilisticOracle):
    """
    Description:
      Constructor

    Parameters:
      input_size: Refer to _MultilayerPerceptron constructor
      n_classes: Refer to _MultilayerPerceptron constructor
      hidden_layers: Refer to _MultilayerPerceptron constructor
      learning_rate: Learning rate to use during training w/ SGD
      momentum: SGD momentum to use during training
      epochs: Number of epochs for training. If None, then validation data
              must be supplied during training to decide when to stop.
    """
    def __init__(self, input_size, n_classes, hidden_layers = [128],
            learning_rate = 0.01, momentum = 0.9, epochs = None):
        super().__init__()
        self.model = _MultilayerPerceptron(input_size, n_classes, hidden_layers)
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                lr = learning_rate, momentum = momentum)
        self.epochs = epochs

    """
    See adtools.evaluation.oracle.oracle.Oracle
    """
    def fit(self, training_data: torch.Tensor,
            training_data_targets: torch.Tensor,
            validation_data: torch.Tensor = None,
            validation_data_targets: torch.Tensor = None, **kwargs) -> None:
        improved = True
        previous_val_loss = float('inf')
        epochs_completed = 0
        training = True
        while training:
            self.model.train()
            predictions = self.model(training_data)
            loss = self.loss_module(predictions, training_data_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epochs_completed += 1
            if self.epochs is not None and epochs_completed >= self.epochs:
                training = False
                break

            if validation_data is not None and \
                    validation_data_targets is not None:
                with torch.no_grad():
                    self.model.eval()
                    predictions = self.model(validation_data)
                    loss = self.loss_module(predictions,
                            validation_data_targets)
                    loss = loss.item()
                    if loss < previous_val_loss:
                        previous_val_loss = loss
                    else:
                        training = False
                        break

    """
    See adtools.evaluation.oracle.oracle.Oracle
    """
    def predict_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(data)
            probabilities = torch.nn.functional.softmax(predictions, dim = 1)
            return probabilities[::, 1]

    def to(self, device: torch.device):
        super().to(device)
        self.model = self.model.to(device)
        return self

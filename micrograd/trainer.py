from micrograd.engine import Value
from micrograd.batch_iterator import BatchIterator
from micrograd.nn import Module
from micrograd.optimizer import Optimizer


History = {}


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model, optimizer, loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, data_iterator, num_epochs=500, verbose=False):
        """Fits the model to the data"""

        history = {"loss": []}
        epoch_loss = 0
        epoch_y_true = []
        epoch_y_pred = []
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            # Reset epoch data
            epoch_loss = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                # Forward pass
                outputs = list(map(self.model, batch[0]))

                batch_y_true = [Value(val) for sublist in batch[1]
                                for val in sublist]
                batch_y_pred = [val for sublist in outputs for val in sublist]
                # Loss computation
                # [item for sublist in outputs[0] for item in sublist]
                batch_loss = self.loss(batch_y_true, batch_y_pred)
                epoch_loss += batch_loss.data

                # Store batch predictions and ground truth for computing epoch metrics
                epoch_y_pred.extend(batch_y_pred)
                epoch_y_true.extend(batch[1])

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            # Accuracy computation for epoch

            # Record training history
            history["loss"].append(epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {epoch_loss:.6f}, "
                )

        return history

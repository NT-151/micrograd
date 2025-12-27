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
            
            self.optimizer.zero_grad()
            
            epoch_loss = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
            
                outputs = list(map(self.model, batch[0]))

                batch_y_true = [Value(val) for sublist in batch[1]
                                for val in sublist]
                batch_y_pred = [val for sublist in outputs for val in sublist]

                batch_loss = self.loss(batch_y_true, batch_y_pred)
                epoch_loss += batch_loss.data

                epoch_y_pred.extend(batch_y_pred)
                epoch_y_true.extend(batch[1])

                batch_loss.backward()
                self.optimizer.step()

            history["loss"].append(epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {epoch_loss:.6f}, "
                )

        return history

history = {}

class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model, optimizer, loss, name=None):
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
            epoch_loss = 0
            num_batches = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                self.optimizer.zero_grad()
                outputs = self.model(batch[0])

                batch_loss = self.loss(batch[1], outputs)
                epoch_loss += batch_loss.data
                num_batches += 1

                epoch_y_pred.extend(outputs)
                epoch_y_true.extend(batch[1])

                batch_loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {avg_epoch_loss:.6f}, "
                )

        return history

    def vae_fit(self, data_iterator, num_epochs=500, verbose=False):
        """Fits the model to the data"""

        history = {"loss": []}
        epoch_loss = 0
        epoch_y_true = []
        epoch_y_pred = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            epoch_y_true = []
            epoch_y_pred = []

            for batch in data_iterator():
                self.optimizer.zero_grad()
                recon, mu, log_var = self.model(batch[0])

                batch_loss = self.loss(recon, batch[1], mu, log_var)
                epoch_loss += batch_loss.data
                num_batches += 1

                epoch_y_pred.extend(recon)
                epoch_y_true.extend(batch[1])

                batch_loss.backward()
                self.optimizer.step()

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_epoch_loss)
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {avg_epoch_loss:.6f}, "
                )

        return history


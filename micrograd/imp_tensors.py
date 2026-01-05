from trainer import Trainer
import random
import matplotlib.pyplot as plt
import numpy as np
from optimizer import SGD
from batch_iterator import BatchIterator
from engine import Value
from nn import AutoEncoder
from sklearn.model_selection import train_test_split
from loss_funcs import mean_squared_error
from utils import mnist_reader
random.seed(1337)
np.random.seed(1337)
np.set_printoptions(suppress=True)



X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0

# reshape to images if you need 2D
X_train_img = X_train.reshape(-1, 28, 28)
X_test_img  = X_test.reshape(-1, 28, 28)

print(X_train.shape, y_train.shape)
print(X_train_img.shape)

# import matplotlib.pyplot as plt

# plt.imshow(X_train_img[0], cmap="gray")
# plt.title(f"label: {y_train[0]}")
# plt.axis("off")
# plt.show()
auto = AutoEncoder(in_embeds=784, hidden_layers=[
                   392, 196], latent_dim=100, act_func=Value.sigmoid)
optimizer = SGD(auto.parameters(), learning_rate=1)
data_iterator = BatchIterator(X_train, X_train, 512)
trainer = Trainer(auto, optimizer, loss=mean_squared_error)

history = trainer.fit(data_iterator, num_epochs=50, verbose=True)

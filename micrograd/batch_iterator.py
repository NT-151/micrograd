"""
Heavily inspired by https://github.com/joelgrus/joelnet/blob/master/joelnet/data.py
"""
import numpy as np
import random
from engine import Value
from nn import AutoEncoder

# Batch = NamedTuple("Batch", [("inputs", List[Vector]), ("targets", Vector)])


class BatchIterator:
    """Iterates on data by batches"""

    def __init__(self, inputs, targets, batch_size=32, shuffle=True):
        self.inputs = inputs.data
        self.targets = targets.data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        starts = list(range(0, len(self.inputs), self.batch_size))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield (Value(batch_inputs), Value(batch_targets))

# hey = Value(np.random.normal(size=(100, 15)))
# auto = AutoEncoder(in_embeds=10, hidden_layers=[
#                    8, 6, 4], latent_dim=2, act_func=Value.sigmoid)

# data_iterator = BatchIterator(hey, hey, 12)

# for batch in data_iterator():
#     auto(batch[0])

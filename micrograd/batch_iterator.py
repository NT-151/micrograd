"""
Heavily inspired by https://github.com/joelgrus/joelnet/blob/master/joelnet/data.py
"""
import random


# Batch = NamedTuple("Batch", [("inputs", List[Vector]), ("targets", Vector)])

class BatchIterator:
    """Iterates on data by batches"""

    def __init__(self, inputs, targets, batch_size=32, shuffle=True):
        self.inputs = inputs
        self.targets = targets
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
            yield (batch_inputs, batch_targets)

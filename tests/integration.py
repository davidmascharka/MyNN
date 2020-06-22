""" 
Train the examples and ensure performance is acceptable. This is intended to be a placeholder to ensure
nothing in this package breaks when MyGrad updates until they're merged.
"""

import gzip
import os
from pathlib import Path
import urllib.request

import numpy as np

from mynn.activations import relu
from mynn.losses import softmax_cross_entropy
from mynn.layers import dense, conv
from mynn.initializers import glorot_uniform
from mynn.optimizers import SGD
from mygrad.nnet.layers.pooling import max_pool


def download_mnist(path=Path("mnist.npz"), server_url="http://yann.lecun.com/exdb/mnist/", tmp_file="__mnist.bin"):
    if Path(path).is_file():
        return None  # already exists

    urls = dict(
        tr_img="train-images-idx3-ubyte.gz",
        tr_lbl="train-labels-idx1-ubyte.gz",
        te_img="t10k-images-idx3-ubyte.gz",
        te_lbl="t10k-labels-idx1-ubyte.gz",
    )

    data = {}
    for type_ in ["tr", "te"]:
        img_key = type_ + "_img"
        lbl_key = type_ + "_lbl"
        print(f"Downloading from: {server_url + urls[img_key]}")
        with urllib.request.urlopen(server_url + urls[img_key]) as response:
            try:
                with open(tmp_file, "wb") as handle:
                    handle.write(response.read())

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=16)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        print(f"Downloading from: {server_url + urls[lbl_key]}")
        with urllib.request.urlopen(server_url + urls[lbl_key]) as response:
            try:
                with open(tmp_file, "wb") as handle:
                    handle.write(response.read())

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp_lbls = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=8)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        data[img_key] = tmp.reshape(tmp_lbls.shape[0], 1, 28, 28)
        data[lbl_key] = tmp_lbls

    print(f"Saving to: {path}")
    with Path(path).open(mode="wb") as f:
        np.savez_compressed(
            f, x_train=data["tr_img"], y_train=data["tr_lbl"], x_test=data["te_img"], y_test=data["te_lbl"]
        )


def load_mnist(fname="mnist.npz"):
    with np.load(fname) as data:
        out = tuple(data[str(key)] for key in ["x_train", "y_train", "x_test", "y_test"])
    print("mnist loaded")
    return out


class ToyData:
    def __init__(self):
        n_train = round(1000 // 1.2)
        n_val = 1000 - n_train
        self._coords = np.zeros((1000 * 3, 2))
        self._labels = np.zeros(1000 * 3, dtype=np.uint8)

        y_labels = np.zeros((1000 * 3, 3), dtype=np.uint8)
        for j in range(3):
            ix = range(1000 * j, 1000 * (j + 1))
            r = np.linspace(0.0, 1, 1000)
            t = np.linspace(0, 2 * np.pi, 1000) + np.random.randn(1000) * 0.2
            t += j / 3 * 2 * np.pi
            self._coords[ix] = np.column_stack((r * np.sin(t), r * np.cos(t)))
            self._labels[ix] = j
            y_labels[ix, j] = 1

        train_ids = np.concatenate(
            [np.random.choice(range(1000 * i, 1000 * (i + 1)), n_train, replace=False) for i in range(3)]
        )
        train_ids = np.random.choice(train_ids, 3 * n_train, replace=False)
        y_ids = np.random.choice(list(set(range(3 * 1000)) - set(train_ids)), 3 * n_val, replace=False)

        self.x_train = self._coords[train_ids,].astype("float32")
        self.y_train = y_labels[train_ids,].astype("float32")

        self.x_test = self._coords[y_ids,].astype("float32")
        self.y_test = y_labels[y_ids,].astype("float32")

    def load_data(self):
        return (self.x_train, np.argmax(self.y_train, axis=-1), self.x_test, np.argmax(self.y_test, axis=-1))


class SpiralModel:
    def __init__(self):
        self.layer1 = dense(2, 100)
        self.layer2 = dense(100, 3)

    def __call__(self, x):
        return self.layer2(relu(self.layer1(x)))

    @property
    def parameters(self):
        return self.layer1.parameters + self.layer2.parameters


def train_epoch(train_data, train_labels, model, optim, batch_size=64):
    idxs = np.arange(len(train_data))
    np.random.shuffle(idxs)

    for batch in range(0, len(idxs), batch_size):
        batch_data = train_data[idxs[batch:batch+batch_size]]
        batch_labels = train_labels[idxs[batch:batch+batch_size]]

        outs = model(batch_data)
        loss = softmax_cross_entropy(outs, batch_labels)
        loss.backward()
        optim.step()
        loss.null_gradients()


def eval_epoch(test_data, test_labels, model, batch_size=64):
    num_correct, num_total = 0, 0
    for batch in range(0, len(test_data), batch_size):
        batch_data = test_data[batch:batch+batch_size]
        batch_labels = test_labels[batch:batch+batch_size]

        outs = model(batch_data)
        outs.null_gradients()

        correct = (outs.data.argmax(axis=1) == batch_labels).sum()
        samples = len(outs)
        num_correct += correct
        num_total += samples

    return num_correct, num_total


def train_spiral():
    toy_data = ToyData()

    train_data, train_labels, test_data, test_labels = toy_data.load_data()

    model = SpiralModel()
    optim = SGD(model.parameters, learning_rate=0.01, momentum=0.99)
    for _ in range(30):
        train_epoch(train_data=train_data, train_labels=train_labels, model=model, optim=optim)

    correct, total = eval_epoch(test_data=test_data, test_labels=test_labels, model=model)
    if correct / total < 0.9:
        raise ValueError(f"Expected spiral accuracy at least 90% but got {correct / total * 100:0.2f}")


class MnistModel:
    def __init__(self):
        init = glorot_uniform
        args = {"gain": np.sqrt(2)}
        self.conv1 = conv(1, 16, 3, 3, padding=1, weight_initializer=init, weight_kwargs=args)
        self.conv2 = conv(16, 16, 3, 3, padding=1, weight_initializer=init, weight_kwargs=args)
        self.conv3 = conv(16, 32, 3, 3, padding=1, weight_initializer=init, weight_kwargs=args)
        self.conv4 = conv(32, 32, 3, 3, padding=1, weight_initializer=init, weight_kwargs=args)
        self.dense = dense(32*7*7, 10)

    def __call__(self, x):
        x = relu(self.conv2(relu(self.conv1(x))))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.conv4(relu(self.conv3(x))))
        x = max_pool(x, (2, 2), 2)
        return self.dense(x.reshape(-1, 32*7*7))

    @property
    def parameters(self):
        params = []
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.dense):
            params += list(layer.parameters)
        return params


def train_mnist():
    download_mnist()
    train_data, train_labels, val_data, val_labels = load_mnist()

    train_data = train_data / 255
    val_data = val_data / 255

    model = MnistModel()
    optim = SGD(model.parameters, learning_rate=0.001, momentum=0.99)

    for epoch in range(5):
        train_epoch(train_data=train_data, train_labels=train_labels, model=model, optim=optim)
    correct, total = eval_epoch(test_data=val_data, test_labels=val_labels, model=model)
    if correct / total < 0.95:
        raise ValueError(f"Expected MNIST accuracy at least 95% but got {correct / total * 100:0.2f}")


if __name__ == "__main__":
    train_spiral()
    train_mnist()

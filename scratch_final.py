print("Project: Scratch Final")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import pickle
import os
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist
import tkinter as tk
from PIL import Image
import cv2


def fetch_data(model, dataset):
    if dataset == "letters":

        train_df = pd.read_csv("EMNIST Kaggle/emnist-balanced-train.csv", header=None)
        test_df = pd.read_csv("EMNIST Kaggle/emnist-balanced-test.csv", header=None)

        y_train = train_df.iloc[:, 0].values
        x_train = train_df.iloc[:, 1:].values

        y_test = test_df.iloc[:, 0].values
        x_test = test_df.iloc[:, 1:].values

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

        x_train = np.flip(np.rot90(x_train, k=1, axes=(2, 3)), axis=2)
        x_test = np.flip(np.rot90(x_test, k=1, axes=(2, 3)), axis=2)

        if model == "fc":
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)
        if model == "conv":
            x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
            x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

        y_train_one_hot = to_one_hot(y_train, 47)
        y_test_one_hot = to_one_hot(y_test, 47)

        return x_train, y_train_one_hot, y_train, x_test, y_test_one_hot, y_test

    else:

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255

        if model == "fc":
            x_train = x_train.reshape(60000, -1)
            x_test = x_test.reshape(10000, -1)
        if model == "conv":
            x_train = x_train.reshape(60000, 1, 28, 28)
            x_test = x_test.reshape(10000, 1, 28, 28)

        y_train_one_hot = to_one_hot(y_train, 10)
        y_test_one_hot = to_one_hot(y_test, 10)

        return x_train, y_train_one_hot, y_train, x_test, y_test_one_hot, y_test

def to_one_hot(labels, n):
    one_hot = np.zeros((labels.size, n))
    one_hot[np.arange(labels.size), labels.astype(int)] = 1  # Modified
    return one_hot


class nn:
    def __init__(self):
        pass

    class Convolve:
        def __init__(self, in_channels, out_channels, kernel_size, padding):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.padding = padding
            self.kernel = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
                           * np.sqrt(2 / (in_channels * kernel_size * kernel_size)))
            self.biases = np.zeros(out_channels)
            self.input_padded = None
            self.dkernel = None
            self.dbiases = None
            self.mkernel = 0
            self.mbiases = 0
            self.vkernel = 0
            self.vbiases = 0
            self.disable_kernel_update = False

        def forward_naive(self, x):
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
            self.input_padded = x_padded
            x_convolved = np.array(
                [[scipy.signal.correlate(image, kernel, mode="valid").squeeze() for kernel in self.kernel] for image in
                 x_padded])
            return x_convolved + self.biases.reshape(1, -1, 1, 1)

        def forward(self, x):
            # pad
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
            self.input_padded = x_padded
            N, C, H, W = x_padded.shape
            D, C, K, K = self.kernel.shape
            im2col = sliding_window_view(x_padded, (C, K, K), (1, 2, 3))
            im2col = im2col.reshape(N, -1, C * K * K)
            kernel2col = self.kernel.reshape(D, -1)
            product = im2col @ kernel2col.T
            x_convolved = product.reshape(N, H - K + 1, W - K + 1, D)
            x_convolved = np.transpose(x_convolved, (0, 3, 1, 2)) + self.biases.reshape(1, -1, 1, 1)
            return x_convolved

        def backward_naive(self, dvalues):  # holup forgor about batches, dvalues is 4D tensor. But parameters are fine
            dvalues_padded = np.pad(dvalues, (
                (0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                (self.kernel_size - 1, self.kernel_size - 1)))
            kernel_modified = np.swapaxes(np.flip(self.kernel, axis=(2, 3)), 0, 1)
            grad = np.array(
                [[scipy.signal.correlate(a, kernel, mode="valid").squeeze() for kernel in kernel_modified] for a in
                 dvalues_padded])
            if self.padding > 0:
                grad = grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

            self.dbiases = np.sum(dvalues, axis=(0, 2, 3)) / dvalues.shape[0]
            self.dkernel = np.zeros_like(self.kernel)

            for n in range(self.input_padded.shape[0]):
                for d in range(self.out_channels):
                    for c in range(self.in_channels):
                        self.dkernel[d][c] += scipy.signal.correlate(self.input_padded[n][c], dvalues[n][d], mode="valid")
            self.dkernel /= self.input_padded.shape[0]

            return grad

        def backward(self, dvalues):
            dvalues_padded = np.pad(dvalues, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)))
            kernel_modified = np.swapaxes(np.flip(self.kernel, axis=(2, 3)), 0, 1)
            N, D, H, W = dvalues_padded.shape
            C, D, K, K = kernel_modified.shape  # CD reversed since flipped axes 0 and 1
            im2col = sliding_window_view(dvalues_padded, (D, K, K), (1, 2, 3))
            im2col = im2col.reshape(N, -1, D * K * K)
            kernel2col = kernel_modified.reshape(C, -1)
            product = im2col @ kernel2col.T
            grad = product.reshape(N, H - K + 1, W - K + 1, C)
            grad = np.transpose(grad, (0, 3, 1, 2))
            if self.padding > 0:
                grad = grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

            self.dbiases = np.sum(dvalues, axis=(0, 2, 3)) / dvalues.shape[0]
            self.dkernel = np.zeros_like(self.kernel)

            N, C, H, W = self.input_padded.shape
            N, D, X, Y = dvalues.shape
            im2col = sliding_window_view(self.input_padded, (1, X, Y), (1, 2, 3))
            im2col = im2col.reshape(N, C * K * K, X * Y)
            dvalues2col = dvalues.reshape(N, D, -1)
            product = im2col @ np.swapaxes(dvalues2col, 1, 2)
            product = product.reshape(N, C, K, K, D)
            product = np.transpose(product, (0, 4, 1, 2, 3))
            self.dkernel = np.mean(product, 0)

            return grad

    class MaxPool:
        def __init__(self, pool_size):
            self.pool_size = pool_size
            self.input_reshaped = None
            self.output = None
            self.excess = None

        def forward(self, x):
            x_trimmed = x[:, :, :x.shape[2] // self.pool_size * self.pool_size,
                        :x.shape[3] // self.pool_size * self.pool_size]
            N, C, H, W = x_trimmed.shape
            x_reshaped = x_trimmed.reshape(N, C, H // self.pool_size, self.pool_size, W // self.pool_size,
                                           self.pool_size)
            self.input_reshaped = x_reshaped
            self.output = x_reshaped.max(axis=(3, 5))
            self.excess = [x.shape[2] % self.pool_size, x.shape[3] % self.pool_size]
            return self.output

        def backward(self, dvalues):
            N, C, H, W = dvalues.shape
            grad_reshaped = dvalues.reshape(N, C, H, 1, W, 1)
            # better to do mask here to avoid excess computation when not training the model
            N, C, H, W = self.output.shape
            output_reshaped = self.output.reshape(N, C, H, 1, W, 1)  # for broadcasting
            mask = (self.input_reshaped == output_reshaped).astype(int)
            grad_unpadded = (mask * grad_reshaped).reshape(N, C, H * self.pool_size, W * self.pool_size)
            grad = np.pad(grad_unpadded, ((0, 0), (0, 0), (0, self.excess[0]), (0, self.excess[1])))
            return grad # EXPERIMENTAL

    class Flatten:
        def __init__(self):
            self.shape = None

        def forward(self, x):
            self.shape = x.shape
            return x.reshape(self.shape[0], -1)

        def backward(self, dvalues):
            return dvalues.reshape(self.shape)

    class Linear:
        def __init__(self, n_in, n_neurons):
            self.weights = np.random.randn(n_in, n_neurons) * np.sqrt(2 / n_in)
            self.biases = np.zeros(n_neurons)
            self.dweights = None
            self.dbiases = None
            self.mweights = 0
            self.mbiases = 0
            self.vweights = 0
            self.vbiases = 0
            self.input = None

        def forward(self, x):
            self.input = x
            return x @ self.weights + self.biases

        def backward(self, dvalues):
            self.dweights = np.dot(self.input.T, dvalues) / self.input.shape[0]
            self.dbiases = np.mean(dvalues, axis=0)
            grad = np.dot(dvalues, self.weights.T)
            return grad

    class ReLu:
        def __init__(self):
            self.input = None

        def forward(self, x):
            self.input = x
            return np.maximum(0, x).copy()

        def backward(self, dvalues):
            grad = dvalues.copy()
            grad[self.input <= 0] = 0
            return grad

    class SoftmaxCrossEntropy:
        def __init__(self):
            self.y_pred = None
            self.y_truth = None

        def forward(self, x, y_truth):
            x = x - np.max(x, axis=1, keepdims=True)
            y = np.exp(x)
            y = y / np.sum(y, axis=1, keepdims=True)
            self.y_pred = y
            y = np.clip(y, 1e-8, 1 - 1e-8)
            sample_loss = -np.sum(y_truth * np.log(y), axis=1)
            loss = np.mean(sample_loss)
            self.y_truth = y_truth
            return self.y_pred, loss

        def backward(self):
            n = self.y_pred.shape[0]
            grad = self.y_pred - self.y_truth
            grad /= n
            return grad


class Adam():

    def __init__(self, lr, beta1, beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8

    def optimize(self, epoch, param, gradient, momentum, variance):
        momentum = self.beta1 * momentum + (1 - self.beta1) * gradient
        variance = self.beta2 * variance + (1 - self.beta2) * (gradient ** 2)
        momentum_adjusted = momentum / (1 - self.beta1 ** epoch)
        variance_adjusted = variance / (1 - self.beta2 ** epoch)
        new_param = param - self.lr * momentum_adjusted / (np.sqrt(variance_adjusted) + self.epsilon)

        return new_param, momentum, variance

class SGD():

    def __init__(self, lr):
        self.lr = lr

class PerceptronModel(nn):

    def __init__(self, outputs, optimizer):
        super().__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, outputs)
        self.final = nn.SoftmaxCrossEntropy()  # nice as object since data should be stored
        self.relu1 = nn.ReLu()
        self.relu2 = nn.ReLu()

        self.optim = optimizer

    def forward(self, x, y_truth):
        x = self.linear1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear2.forward(x)
        x = self.relu2.forward(x)
        x = self.linear3.forward(x)
        x = self.final.forward(x, y_truth)
        return x

    def backward(self):
        grad = self.final.backward()
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)

    def optimizer_step(self, epoch):
        if isinstance(self.optim, SGD):
            self.simple_sgd(self.optim.lr)
        if isinstance(self.optim, Adam):
            self.optim_adam(epoch)

    def simple_sgd(self, learning_rate):
        # ("SGD Perceptron")
        self.backward()
        self.linear1.weights -= learning_rate * self.linear1.dweights
        self.linear1.biases -= learning_rate * self.linear1.dbiases
        self.linear2.weights -= learning_rate * self.linear2.dweights
        self.linear2.biases -= learning_rate * self.linear2.dbiases
        self.linear3.weights -= learning_rate * self.linear3.dweights
        self.linear3.biases -= learning_rate * self.linear3.dbiases

    def optim_adam(self, epoch):
        self.backward()

        self.linear1.weights, self.linear1.mweights, self.linear1.vweights = self.optim.optimize(
            epoch, self.linear1.weights, self.linear1.dweights, self.linear1.mweights, self.linear1.vweights)
        self.linear1.biases, self.linear1.mbiases, self.linear1.vbiases = self.optim.optimize(
            epoch, self.linear1.biases, self.linear1.dbiases, self.linear1.mbiases, self.linear1.vbiases)
        self.linear2.weights, self.linear2.mweights, self.linear2.vweights = self.optim.optimize(
            epoch, self.linear2.weights, self.linear2.dweights, self.linear2.mweights, self.linear2.vweights)
        self.linear2.biases, self.linear2.mbiases, self.linear2.vbiases = self.optim.optimize(
            epoch, self.linear2.biases, self.linear2.dbiases, self.linear2.mbiases, self.linear2.vbiases)
        self.linear3.weights, self.linear3.mweights, self.linear3.vweights = self.optim.optimize(
            epoch, self.linear3.weights, self.linear3.dweights, self.linear3.mweights, self.linear3.vweights)
        self.linear3.biases, self.linear3.mbiases, self.linear3.vbiases = self.optim.optimize(
            epoch, self.linear3.biases, self.linear3.dbiases, self.linear3.mbiases, self.linear3.vbiases)


class ConvModel(nn):  # 16 32 1568 128

    def __init__(self, outputs, optimizer):
        super().__init__()
        self.conv1 = nn.Convolve(1, 16, 3, 1)
        self.conv2 = nn.Convolve(16, 32, 3, 1)
        self.pool1 = nn.MaxPool(2)
        self.pool2 = nn.MaxPool(2)
        self.linear1 = nn.Linear(1568, 128)
        self.linear2 = nn.Linear(128, outputs)
        self.final = nn.SoftmaxCrossEntropy()
        self.relu1 = nn.ReLu()
        self.relu2 = nn.ReLu()
        self.relu3 = nn.ReLu()
        self.flatten = nn.Flatten()

        self.optim = optimizer

    def forward(self, x, y_truth):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.relu3.forward(x)
        x = self.linear2.forward(x)
        x = self.final.forward(x, y_truth)
        return x

    def backward(self):
        # print("BACKWARDS")
        grad = self.final.backward()
        grad = self.linear2.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.linear1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def optimizer_step(self, epoch):
        if isinstance(self.optim, SGD):
            self.simple_sgd(self.optim.lr)
        if isinstance(self.optim, Adam):
            self.optim_adam(epoch)

    def simple_sgd(self, learning_rate):
        # print("SGD Conv")
        self.backward()
        self.linear1.weights -= learning_rate * self.linear1.dweights
        self.linear1.biases -= learning_rate * self.linear1.dbiases
        self.linear2.weights -= learning_rate * self.linear2.dweights
        self.linear2.biases -= learning_rate * self.linear2.dbiases
        self.conv1.kernel -= learning_rate * self.conv1.dkernel
        self.conv1.biases -= learning_rate * self.conv1.dbiases
        self.conv2.kernel -= learning_rate * self.conv2.dkernel
        self.conv2.biases -= learning_rate * self.conv2.dbiases

    def optim_adam(self, epoch):
        self.backward()

        self.linear1.weights, self.linear1.mweights, self.linear1.vweights = self.optim.optimize(
            epoch, self.linear1.weights, self.linear1.dweights, self.linear1.mweights, self.linear1.vweights)
        self.linear1.biases, self.linear1.mbiases, self.linear1.vbiases = self.optim.optimize(
            epoch, self.linear1.biases, self.linear1.dbiases, self.linear1.mbiases, self.linear1.vbiases)
        self.linear2.weights, self.linear2.mweights, self.linear2.vweights = self.optim.optimize(
            epoch, self.linear2.weights, self.linear2.dweights, self.linear2.mweights, self.linear2.vweights)
        self.linear2.biases, self.linear2.mbiases, self.linear2.vbiases = self.optim.optimize(
            epoch, self.linear2.biases, self.linear2.dbiases, self.linear2.mbiases, self.linear2.vbiases)

        self.conv1.kernel, self.conv1.mkernel, self.conv1.vkernel = self.optim.optimize(
            epoch, self.conv1.kernel, self.conv1.dkernel, self.conv1.mkernel, self.conv1.vkernel)
        self.conv1.biases, self.conv1.mbiases, self.conv1.vbiases = self.optim.optimize(
            epoch, self.conv1.biases, self.conv1.dbiases, self.conv1.mbiases, self.conv1.vbiases)
        self.conv2.kernel, self.conv2.mkernel, self.conv2.vkernel = self.optim.optimize(
            epoch, self.conv2.kernel, self.conv2.dkernel, self.conv2.mkernel, self.conv2.vkernel)
        self.conv2.biases, self.conv2.mbiases, self.conv2.vbiases = self.optim.optimize(
            epoch, self.conv2.biases, self.conv2.dbiases, self.conv2.mbiases, self.conv2.vbiases)

def evaluate_model(model, dataset, batch_sizes, epochs):
    if isinstance(model, PerceptronModel):
        model_type = "fc"
    else:
        model_type = "conv"

    x_train, y_train_one_hot, y_train, x_test, y_test_one_hot, y_test = fetch_data(model_type, dataset)

    start = time.time()
    times = [start]

    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        x_shuffled = x_train[perm]
        y_shuffled = y_train_one_hot[perm]
        for i in range(x_train.shape[0] // batch_sizes):
            x = x_shuffled[i * batch_sizes: (i + 1) * batch_sizes]
            y = y_shuffled[i * batch_sizes: (i + 1) * batch_sizes]
            loss = model.forward(x, y)[1]
            if i % 100 == 0:
                print(f"Training | Epoch: {epoch}, Batch: {i}, Loss: {loss}")
            model.optimizer_step(epoch+1)
        times.append(time.time())

    n_correct = 0
    for i in range(x_test.shape[0] // batch_sizes):
        x = x_test[i * batch_sizes: (i + 1) * batch_sizes]
        y_one_hot = y_test_one_hot[i * batch_sizes: (i + 1) * batch_sizes]
        y = y_test[i * batch_sizes: (i + 1) * batch_sizes]

        pred = np.argmax(model.forward(x, y_one_hot)[0], axis=1)
        n_correct += sum(pred == y)

        if i % 10000 == 0:
            print(f"Evaluating Performance | Batch: {i}")

    accuracy_test = n_correct / x_test.shape[0]

    times = np.array(times) - start
    times = np.diff(times)
    mean_training_time = np.mean(times)
    std_training_time = np.std(times)

    n_correct = 0
    for i in range(x_train.shape[0] // batch_sizes):
        x = x_train[i * batch_sizes: (i + 1) * batch_sizes]
        y_one_hot = y_train_one_hot[i * batch_sizes: (i + 1) * batch_sizes]
        y = y_train[i * batch_sizes: (i + 1) * batch_sizes]

        pred = np.argmax(model.forward(x, y_one_hot)[0], axis=1)
        n_correct += sum(pred == y)

        if i % 10000 == 0:
            print(f"Testing Memorization | Batch: {i}")

    accuracy_train = n_correct / x_train.shape[0]

    print(f"Final Report | Accuracy: {accuracy_test * 100:.3f}% | Memorization: {accuracy_train * 100:.3f}% || "
          f"Training Time/Epoch (mean/std): {mean_training_time:.3f}s / {std_training_time:.3f}s")

def draw_window():
    # Window size: roughly 1/8 of 1440p screen (~360x202)
    window_size = 360
    brush_size = 12

    # Tkinter setup
    root = tk.Tk()
    root.title("Draw Window")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=window_size, height=window_size, bg="black")
    canvas.grid(row=0, column=0, columnspan=2)

    # Image buffer (grayscale 0–255)
    img_array = np.zeros((window_size, window_size), dtype=np.uint8)

    # Drawing behavior
    def draw(event):
        x, y = event.x, event.y
        # Draw white circle on canvas
        canvas.create_oval(x - brush_size, y - brush_size, x + brush_size, y + brush_size,
                           fill="white", outline="white")
        # Paint in array
        x_min, x_max = max(0, x - brush_size), min(window_size, x + brush_size)
        y_min, y_max = max(0, y - brush_size), min(window_size, y + brush_size)
        img_array[y_min:y_max, x_min:x_max] = 255

    canvas.bind("<B1-Motion>", draw)

    # Reset function
    def reset():
        canvas.delete("all")
        canvas.create_rectangle(0, 0, window_size, window_size, fill="black", outline="black")
        img_array[:] = 0

    reset_btn = tk.Button(root, text="Reset", command=reset)
    reset_btn.grid(row=1, column=0, sticky="ew")

    # Submit function
    submitted = {"done": False, "image": None}

    def submit():
        submitted["image"] = img_array.copy()
        submitted["done"] = True
        root.destroy()

    submit_btn = tk.Button(root, text="Submit", command=submit)
    submit_btn.grid(row=1, column=1, sticky="ew")

    root.mainloop()

    if not submitted["done"]:
        return None

    # Resize to 28x28
    img_pil = Image.fromarray(submitted["image"])
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    return np.array(img_resized, dtype=np.uint8)

def load_emnist_mapping(path="emnist-balanced-mapping.txt"):
    mapping = pd.read_csv(path, sep=' ', header=None, names=['label', 'ascii'])
    label_to_char = {row.label: chr(row.ascii) for _, row in mapping.iterrows()}
    return label_to_char

def manual_test_model(model, dataset):
    image = draw_window()

    coords = cv2.findNonZero(image)  # find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)
    digit = image[y:y + h, x:x + w]
    digit = cv2.resize(digit, (24, 24))  # resize bounding box to 20x20

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 24) // 2
    y_offset = (28 - 24) // 2
    canvas[y_offset:y_offset + 24, x_offset:x_offset + 24] = digit


    if isinstance(model, ConvModel):
        image = canvas.reshape(1, 1, 28, 28) / 255.0
    else:
        image = canvas.reshape(1, 1, 28*28) / 255.0

    if dataset == "numbers":
        y_pred = model.forward(image, to_one_hot(np.zeros(1), 10))[0]
        print(np.argmax(y_pred))
    else:
        y_pred = model.forward(image, to_one_hot(np.zeros(1), 47))[0]
        mapping = load_emnist_mapping("EMNIST Kaggle/emnist-balanced-mapping.txt")
        print(mapping[np.argmax(y_pred)])


mode = int(input("""Would you like to: 
1. train a model 
2. test a pre-trained model
> """))

if mode == 1:
    model_type = int(input("""Please select the model you wish to train: 
1. Multilayer Perceptron
2. Convolutional Neural Network
> """))
    model_task = int(input("""Please select the dataset the model should learn: 
1. MNIST (digits 0-9)
2. EMNIST (digits, uppercase and lowercase letters)
> """))
    optimizer_type = int(input("""Please select the type of optimizer that should be used: 
1. Stochastic Gradient Descent
2. Adam
> """))
    print("Hyperparameters will now be requested")
    epochs = int(input("Epochs:  "))
    batch_size = int(input("Batch Size:  "))
    learning_rate = float(input("Learning Rate:  "))
    print("Training initiated")

    if optimizer_type == 1:
        optimizer = SGD(learning_rate)
    elif optimizer_type == 2:
        optimizer = Adam(learning_rate, 0.9, 0.999)

    outputs = 0
    if model_task == 1:
        outputs = 10
        dataset = "numbers"
    else:
        outputs = 47
        dataset = "letters"

    if model_type == 1:
        model = PerceptronModel(outputs, optimizer)
    else:
        model = ConvModel(outputs, optimizer)

    evaluate_model(model, dataset, batch_size, epochs)

    procedure = int(input("""Would you like to: 
1. test your model manually
2. quit the application
> """))

    if procedure == 1:
        print("Please open the drawing window.")
        while True:
            manual_test_model(model, dataset)


if mode == 2:
    model_type = int(input("""Please select the model you wish to test: 
1. Multilayer Perceptron | SGD trained for 100 Epochs
2. Multilayer Perceptron | Adam trained for 10 Epochs
3. Convolutional Neural Network | SGD trained for 50 Epochs
4. Convolutional Neural Network | Adam trained for 10 Epochs
> """))
    model_task = int(input("""Please select the dataset the model is trained off: 
1. MNIST (digits 0-9)
2. EMNIST (digits, uppercase and lowercase letters)
> """))

    if model_type == 1:
        if model_task == 1:
            model = "Models/PerceptronModel, MNIST, SGD, 100.pkl"
        elif model_task == 2:
            model = "Models/PerceptronModel, EMNIST, SGD, 100.pkl"
    if model_type == 2:
        if model_task == 1:
            model = "Models/PerceptronModel, MNIST, Adam, 10.pkl"
        elif model_task == 2:
            model = "Models/PerceptronModel, EMNIST, Adam, 10.pkl"
    if model_type == 3:
        if model_task == 1:
            model = "Models/ConvModel, MNIST, SGD, 50.pkl"
        elif model_task == 2:
            model = "Models/ConvModel, EMNIST, SGD, 50.pkl"
    if model_type == 4:
        if model_task == 1:
            model = "Models/ConvModel, MNIST, Adam, 10.pkl"
        elif model_task == 2:
            model = "Models/ConvModel, EMNIST, Adam, 10.pkl"

    with open(model, "rb") as f:
        model = pickle.load(f)

    print("Model loaded. Please open the drawing window.")
    if model_task == 1:
        outputs = 10
        dataset = "numbers"
    else:
        outputs = 47
        dataset = "letters"

    while True:
        manual_test_model(model, dataset)

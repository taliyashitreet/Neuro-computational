import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn_som.som import SOM


class Kohonen:
    def __init__(self, data: np.ndarray, net_size: int, rang):
        """
        SOM: self organization map - weight of each neuron
        :param data: train_data
        :param net_size:
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (h, h, 2)).astype(float) / 1000
        self.data = data
        self.net_size = net_size
        self.rang = rang

    def find_BMU(self, sample):
        """
        find the most close neuron for this sample
        clac the oclid distance from this sample to all neurons,
        pick the neuron that minimize the dist
        :param sample:
        :return:
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    # Update the weights of the SOM cells when given a single training example
    # and the model parameters along with BMU coordinates as a tuple
    def update_weights(self, sample, learn_rate, radius_sq,
                       bmu_idx, step=3):
        """
        update weight of BMU and its neighboors
        :param sample: single training example
        :param learn_rate:
        :param radius_sq:
        :param bmu_idx: the best neuron
        :param step:
        :return:
        """
        x, y = bmu_idx
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.SOM[x, y, :] += learn_rate * (sample - self.SOM[x, y, :])
            return self.SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            for j in range(max(0, y - step), min(self.SOM.shape[1], y + step)):
                dist_sq = np.square(i - x) + np.square(j - y)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (sample - self.SOM[i, j, :])
        return self.SOM

    # Main routine for training an SOM. It requires an initialized SOM grid
    # or a partially trained grid as parameter
    def train_SOM(self, learn_rate=.9, radius_sq=1,
                  lr_decay=.1, radius_decay=.1, epochs=10):
        rand = np.random.RandomState(0)
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(self.data)
            for train_ex in self.data:
                g, h = self.find_BMU(train_ex)
                self.SOM = self.update_weights(train_ex,
                                               learn_rate, radius_sq, (g, h))
            self.plot("curr iter: " + str(epoch) + " , learning rate: " + str(round(learn_rate, 3)),self.rang)
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM

    def plot(self, s ,rang):
        re_wx = self.SOM[:, :, 0]
        re_wy = self.SOM[:, :, 1]

        fig, ax = plt.subplots()
        from_= rang[0]
        till = rang[1]
        ax.set_xlim(from_, till)
        ax.set_ylim(from_, till)
        for i in range(self.SOM.shape[0]):
            xs = []
            ys = []
            xh = []
            yh = []
            for j in range(self.SOM.shape[1]):
                xs.append(self.SOM[i, j, 0])
                ys.append(self.SOM[i, j, 1])
                xh.append(self.SOM[j, i, 0])
                yh.append(self.SOM[j, i, 1])

            ax.plot(xs, ys, 'r-', markersize=0, linewidth=0.7)
            ax.plot(xh, yh, 'r-', markersize=0, linewidth=0.7)

        ax.plot(re_wx, re_wy, color='b', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="c", alpha=0.2)
        plt.title(s)
        plt.show()

class Kohonen_1D:
    def __init__(self, data: np.ndarray, net_size: int):
        """
        SOM: self organization map - weight of each neuron
        :param data: train_data
        :param net_size: num of neurons
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (net_size, 2)).astype(float) / 1000
        self.data = data
        self.net_size = net_size

    def find_BMU(self, sample):
        """
        find the most close neuron for this sample
        clac the oclid distance from this sample to all neurons,
        pick the neuron that minimize the dist
        :param sample: single training example
        :return:
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=1)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def update_weights(self, sample, learn_rate, radius_sq, bmu_idx, step=3):
        """
        update weight of BMU and its neighboors
        :param sample: single training example
        :param learn_rate:
        :param radius_sq:
        :param bmu_idx: the best neuron
        :param step:
        :return:
        """
        x = bmu_idx[0]
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.SOM[x, :] += learn_rate * (sample - self.SOM[x, :])
            return self.SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            dist_sq = np.square(i - x)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            self.SOM[i, :] += learn_rate * dist_func * (sample - self.SOM[i, :])
        return self.SOM

    def train_SOM(self, learn_rate=.9, radius_sq=1,
                  lr_decay=.1, radius_decay=.1, epochs=29):
        """
        train SOM model - for each sample:
            1. find BMU
            2. update weights
            3. update learning rate
            4. update radius
        :param lr_decay: Rate of decay of the learn_rate
        :param radius_decay: Rate of decay of the radius
        :param epochs: num of iteration
        :return:
        """
        rand = np.random.RandomState(0)
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(self.data)
            for sample in self.data:
                x = self.find_BMU(sample)
                self.SOM = self.update_weights(sample,
                                               learn_rate, radius_sq, x)
            self.plot("curr iter: " + str(epoch) + " , learning rate: " + str(round(learn_rate, 3)))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM

    def plot(self, title):
        X = self.SOM[:, 0]  # The X of each point
        Y = self.SOM[:, 1]  # The Y of each point

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        xs = []  # x of each point in axis 1 (cols)
        ys = []  # y of each point in axis 1 (rows)
        for i in range(self.SOM.shape[0]):
            xs.append(self.SOM[i, 0])
            ys.append(self.SOM[i, 1])

        ax.plot(xs, ys, 'r-', markersize=0, linewidth=0.7)
        ax.plot(X, Y, color='b', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="c", alpha=0.2)
        plt.title(title)
        plt.show()


def createData(size: int, part=1):
    if part == 1:
        rand = np.random.RandomState(0)
        data = rand.randint(0, 1000, (size, 2)) / 1000

    elif part == 2:
        rand = np.random.RandomState(0)
        data1 = rand.randint(0, 500, (int(size * 0.8), 2)) / 1000
        data2 = rand.randint(501, 1000, (int(size * 0.2), 2)) / 1000
        data = np.vstack((data1, data2))
        rand.shuffle(data)

    else:
        data = np.ndarray((size, 2))
        s = 0
        while s < size:
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            if 2 <= ((x ** 2) + (y ** 2)) <= 4:
                data[s, 0] = x
                data[s, 1] = y
                s += 1

    return data


def createDataB(part=1):
    data = np.zeros((1500, 2))
    random.seed(11)
    if part == 1:
        Mask = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])

    else:
        Mask = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])

    Mask = np.flip(Mask).T

    n = 0
    random.seed(1)
    while n < 1500:
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        i = int(x)
        j = int(y)
        if Mask[i, j] == 1:
            data[n, 0] = x / 20
            data[n, 1] = y / 20
            n += 1

    return data


if __name__ == '__main__':
    # data = createData(1000, 3)
    # model = Kohonen(data, 100)
    # model.train_SOM()
    #data = createData(1000, 2)
    data=createDataB(2)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()

    model = Kohonen(data, 225,[0,1])
    model.train_SOM()

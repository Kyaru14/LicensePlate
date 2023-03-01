import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

class Dataset(data.Dataset):
    def __init__(self, tau=4, seed: int = None):
        self.x = np.sin(0.01 * np.arange(0, 1000))
        if seed is not None:
            np.random.seed(seed)
        self.x += 0.2 * np.random.randn(len(self.x))
        self.x = torch.tensor(self.x, dtype=torch.float)
        length = len(self.x) - tau
        self.feature = torch.zeros(size=(length, tau))
        self.label = torch.zeros(size=(length, 1))
        for i in range(0, length):
            self.feature[i] = self.x[i:i+tau]
            self.label[i] = self.x[i+tau]

    def __getitem__(self, item):
        return self.feature[item], self.label[item]

    def __len__(self):
        return len(self.label)

class Net(nn.Sequential):
    def __init__(self, tau=4):
        args = [
            nn.Linear(tau, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
                ]
        super().__init__(*args)


if __name__ == '__main__':
    batch_size = 64
    num_workers = 0
    num_epochs = 100
    lr = 0.001
    tau = 200
    train_dataset = Dataset(tau=tau, seed=1)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    net = Net(tau)
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    for epoch in range(num_epochs):
        print(f'epoch {epoch+1}/{num_epochs}')
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
        print('loss:', loss)
    plt.plot(train_dataset.x)
    x_pred = torch.zeros(size=(len(train_dataset.x),), requires_grad=False)
    for i in range(len(x_pred) - tau):
        feature = train_dataset.x[i:i+tau]
        x_pred[i+tau] = net(feature)
    plt.plot(x_pred.detach())
    x_pred2 = torch.zeros(size=(2201,), requires_grad=False)
    feature = train_dataset.x[0:tau]
    for i in range(0, 2000):
        y = net(feature)
        feature = torch.cat([feature[1:], y])
        x_pred2[i + tau] = y
    plt.plot(x_pred2.detach())
    plt.show()

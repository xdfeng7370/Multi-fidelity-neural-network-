import torch
import numpy as np
from torch import nn, optim
from matplotlib import pyplot as plt
class NN1(nn.Module):

    def __init__(self, in_N, m, out_N):
        super(NN1, self).__init__()
        self.m = m
        self.in_N = in_N
        self.out_N = out_N
        self.L1 = nn.Linear(in_N, m)
        self.L2 = nn.Linear(m, m)
        self.L3 = nn.Linear(m, m)
        self.L4 = nn.Linear(m, m)
        self.L5 = nn.Linear(m, out_N)

    def forward(self, x):
        x1 = torch.tanh(self.L1(x))
        x2 = torch.tanh(self.L2(x1))
        x3 = torch.tanh(self.L3(x2))
        x4 = torch.tanh(self.L4(x3))
        x5 = self.L5(x4)
        return x5


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def main():
    model1 = NN1(2, 20, 1)
    model1.apply(weights_init)
    print(model1)
    optimizer = optim.AdamW(model1.parameters(), lr=5e-4)
    nIter = 10000
    training_data = np.load('dataset_low_high.npy')
    training_label = np.load('dataset_high.npy')
    training_label = training_label[:, 1:2]
    loss_value = 1
    it = 0
    while loss_value > 3e-5:
        pred = model1(torch.from_numpy(training_data).float())
        loss = torch.mean(torch.square(torch.from_numpy(training_label).float() - pred))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_value = loss.item()
        if it % 1000 == 0:
            print('It:', it, 'Loss:', loss.item())
        it = it + 1
    torch.save(model1.state_dict(), 'ode_1')

    model1.load_state_dict(torch.load('ode_1'))
    with torch.no_grad():
        data_low = np.load('dataset_low.npy')
        pred_low = model1(torch.from_numpy(data_low).float())
        data_low = torch.from_numpy(data_low).float()
        new_low = torch.cat((data_low[:, 0:1], pred_low), 1)
        data_high = np.load('dataset_high.npy')
        new_training_set = torch.cat((new_low, torch.from_numpy(data_high).float()), 0)  # all high fidelity data
        print(new_training_set.shape)

    model2 = NN1(1, 20, 1)
    model2.apply(weights_init)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    it = 0
    loss_value = 1
    while loss_value > 1e-4:
        pred = model2(new_training_set[:, 0: 1])
        loss2 = torch.mean(torch.square(pred - new_training_set[:, 1: 2]))

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        loss_value = loss2.item()
        if it % 100 == 0:
            print('It:', it, 'Loss:', loss2.item())
        it = it + 1
    torch.save(model2.state_dict(), 'ode_2')
    model2.load_state_dict(torch.load('ode_2'))
    new_data = np.linspace(-1, 1, 501).reshape(-1, 1)
    pred = model2(torch.from_numpy(new_data).float())
    pred = pred.detach().numpy()
    data1 = np.load('dataset_low.npy')
    convert_low = model2(torch.from_numpy(data1[:, 0: 1]).float()).detach().numpy()
    data2 = np.load('dataset_high.npy')
    data3 = np.load('standard_data_high.npy')
    fig, ax = plt.subplots()
    ax.plot(new_data, pred, 'b--', label='$\mathcal{N}_{hf}$')
    plt.scatter(data1[:, 0:1], convert_low, color='', edgecolor='blue', marker='o', label='converted low-fidelity data')
    plt.scatter(data2[:, 0:1], data2[:, 1:2], color='red', marker='x', label='high-fidelity data')
    ax.plot(data3[:, 0: 1], data3[:, 1: 2], 'black', label='Exact')
    plt.xlabel('y')
    plt.ylabel('q(y)')
    plt.legend()
    plt.show()

    y = torch.rand([int(1.35*10**5), 1]) * 2 - 1
    result = model2(y.float())
    result = torch.mean(result).detach().numpy()
    np.save('NN_result.npy', result)
    print(result)
if __name__ == '__main__':
    main()

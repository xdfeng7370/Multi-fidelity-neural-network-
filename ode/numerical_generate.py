import numpy as np
from matplotlib import pyplot as plt
def runge_kuta(u0, a, b, h, y):

    """
    :param u0: initial value at time a
    :param a: initial time
    :param b: termination time
    :param h: step length
    :param y: a hyper parameter
    :return: the solution at time b
    """
    iteration = int((b - a) / h)
    u = u0
    for i in range(iteration):
        t = a + i * h
        f_value = function_f(t, y, u)
        # three second-order Runge-kuta methods
        # u = u + h * function_f(t+h/2, y,  u+f_value*h/2)  # 中点公式
        u = u + h / 2 * (f_value + function_f(t+h, y, u+h*f_value))  # 改进显式 Euler 公式
        # u = u + h/4 * (f_value + 3 * function_f(t+2/3*h, y, u+2*h/3*f_value))  # Heun 公式
    return u

def function_f(t, y, u):
    return 0.25 + np.sin(12*y) + 3 * np.sin(2*t) * np.sin(10*y) * (1+2*y**2) +\
           12 * np.cos(2*t) * np.sin(10*y) * (1+2*y**2) - 0.5 * u

def main():
    y = np.random.rand(int(1.35*10**5), 1) * 2 - 1
    result = np.zeros_like(y)
    for j in range(int(len(y))):
        temp = y[j]
        u0 = 0.5 + 2 * np.sin(12 * temp)
        result[j] = runge_kuta(u0, 0, 100, 0.5, temp)
        if j % 100 == 0:
            print('It:', j)
    result = np.mean(result)
    np.save('numerical_result.npy', result)
    #
    # # y = 1/2  # parameter, y \in [-1, 1]
    # y_low_set = np.linspace(-1, 1, 241)
    # y_high_set = np.linspace(-1, 1, 241)
    # u_low = np.zeros_like(y_low_set)
    # u_high = np.zeros_like(y_high_set)
    # for j in range(int(y_low_set.size)):
    #     y = y_low_set[j]
    #     u0 = 0.5 + 2 * np.sin(12 * y)
    #     u_low[j] = runge_kuta(u0, 0, 100, 0.5, y)
    #
    # for j in range(int(y_high_set.size)):
    #     y = y_high_set[j]
    #     u0 = 0.5 + 2 * np.sin(12 * y)
    #     u_high[j] = runge_kuta(u0, 0, 100, 0.1, y)
    #
    # data_low = np.zeros([180])
    # y_low = np.zeros([180])
    # data_low_high = np.zeros([61])
    # data_high = np.zeros([61])
    # y_high = np.linspace(-1, 1, 61)
    # for i in range(61):
    #     data_high[i] = u_high[4 * i]
    # dataset_high = np.vstack((y_high, data_high)).T
    # j = 0
    # for i in range(241):
    #     if i % 4 != 0:
    #         data_low[j] = u_low[i]
    #         y_low[j] = y_low_set[i]
    #         j = j + 1
    # for i in range(61):
    #     data_low_high[i] = u_low[4 * i]
    # dataset_low_high = np.vstack((y_high, data_low_high)).T
    # dataset_low = np.vstack((y_low, data_low)).T
    # standard_data_high = np.vstack((y_high_set, u_high)).T
    # print(dataset_high.shape, dataset_low.shape)
    # np.save('dataset_low.npy', dataset_low)
    # np.save('dataset_high.npy', dataset_high)
    # np.save('dataset_low_high.npy', dataset_low_high)
    # np.save('standard_data_high.npy', standard_data_high)
    # plt.plot(y_low_set, u_low, 'b', label='$q_{LF}$')
    # plt.plot(y_high_set, u_high, 'fuchsia', label='$q_{HF}$')
    # plt.scatter(dataset_high[:, 0:1], dataset_high[:, 1: 2], color='fuchsia', marker='.', label='high-fidelity data')
    # plt.scatter(dataset_low[:, 0:1], dataset_low[:, 1: 2], color='', edgecolors='blue', marker='v', label='low-fidelity data')
    # plt.xlabel('y')
    # plt.ylabel('q(y)')
    # plt.legend()
    # plt.show()
    #
    # y = np.random.rand([1.35*10**5, 1])
    # result = np.zeros_like(y)
    # for j in range(int(len(y))):
    #     temp = y[j]
    #     u0 = 0.5 + 2 * np.sin(12 * temp)
    #     result[j] = runge_kuta(u0, 0, 100, 0.5, temp)
    # result = np.mean(result)
    # np.save('numerical_result.npy', result)
if __name__ == '__main__':
    main()

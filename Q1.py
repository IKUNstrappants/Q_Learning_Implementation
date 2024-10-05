import numpy as np
import matplotlib.pyplot as plt

def power_matrix(x, power):
    mat = []
    for i in range(power+1):
        mat.append(np.power(x, i)[:, np.newaxis])
    return np.concatenate(mat, axis=1)

def err(x_test, y_test, power):
    X = power_matrix(x, power)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    X_test = power_matrix(x_test, power)
    return np.sum(np.square(X_test @ theta - y_test))

x = np.array([7.2, 1.4, 3.7, 4.8, 4.1, 6.1, 8.1, 5.6, 7.3, 8.2])
y = np.array([0.8, 1.2, 0.6,-0.5,-0.1, 0.4, 1.6, 0.3, 1.9, 1.4])

x_mean = np.mean(x)
y_mean = np.mean(y)

# print(x_mean, y_mean)

theta1 = np.sum((x-x_mean)*(y - y_mean)) / np.sum(np.square(x-x_mean))
theta0 = y_mean - theta1 * x_mean

# print(theta1, theta0)

# print(train[0] * theta1 + theta0 - train[1])

x_plot = np.arange(1, 8.5, 0.1)
plt.scatter(x, y, label="train set")
plt.plot(x_plot, x_plot * theta1-theta0, "orange", label="linear")
# plt.show()

x2 = np.square(x)
# print(x2)

X2 = np.concatenate((np.ones((10, 1)), x[:, np.newaxis], x2[:, np.newaxis]), axis=1)
# print(X2)
theta = np.linalg.inv(X2.T @ X2) @ X2.T @ y
print(theta)
X_plot = np.concatenate((np.ones((len(x_plot), 1)), x_plot[:, np.newaxis], np.square(x_plot)[:, np.newaxis]), axis=1)
plt.plot(x_plot, X_plot @ theta, "green", label="with x2")

X4 = np.concatenate((np.ones((10, 1)),
                     x[:, np.newaxis],
                     x2[:, np.newaxis],
                     np.power(x, 3)[:, np.newaxis],
                     np.power(x, 4)[:, np.newaxis],
                     ), axis=1)

theta = np.linalg.inv(X4.T @ X4) @ X4.T @ y
print(theta)

X_plot = np.concatenate((np.ones((len(x_plot), 1)),
                     x_plot[:, np.newaxis],
                     np.power(x_plot, 2)[:, np.newaxis],
                     np.power(x_plot, 3)[:, np.newaxis],
                     np.power(x_plot, 4)[:, np.newaxis],
                     ), axis=1)
plt.plot(x_plot, X_plot @ theta, "red", label="with x2, x3, x4")

x_test = np.array([7.1, 1.2, 3.8, 4.6, 4.1, 6.5, 8.0, 5.4, 7.2, 8.1])
y_test = np.array([0.7, 1.2, 0.5,-0.5,-0.2, 0.4, 1.6, 0.4, 1.8, 1.2])

plt.scatter(x_test, y_test, label="test set")

plt.legend()
plt.show()

print(err(x_test, y_test, 1), err(x_test, y_test, 2), err(x_test, y_test, 4))
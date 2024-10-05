import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0.31, 0.33, 0.36, 0.61, 0.79, 0.62],
              [0.78, 0.51, 0.73, 0.85, 0.76, 0.93]]).T
y = np.array([[0, 0, 0, 1, 1, 1]]).T

x_test = np.array([[0.72, 0.75, 0.39, 0.82, 0.76, 0.84],
                   [0.38, 0.31, 0.76, 0.77, 0.87, 0.44]]).T
y_test = np.array([[1, 1, 0, 1, 0, 1]]).T

class linear_model():
    def __init__(self, use_random=False, lr=0.1):
        self.theta = np.array([[-1., 1.5, 0.5]]).T if not use_random else np.random.random((3, 1))
        self.lr = lr

    def load_training_set(self, x, y):
        self.x = np.concatenate((np.ones((len(x[:, 0]), 1)), x), axis=1)
        # print(self.x)
        self.y = y

    def iter(self, print_gradient=False):
        # print((1 + np.exp(-1 * self.x @ self.theta)) - self.y)
        # print(((1 + np.exp(-1 * self.x @ self.theta)) - self.y) * self.x)
        print(self.x, self.theta)
        gradient = np.sum((1 / (1 + np.exp(-1 * self.x @ self.theta)) - self.y) * self.x, axis=0) / self.x.shape[0]
        if print_gradient: print(gradient)
        self.theta = self.theta - gradient[:, np.newaxis] * self.lr
    def print_model(self):
        print(f"theta = {self.theta.flatten()}")

    def predict(self, x, y, print_result = True):
        score = np.concatenate((np.ones((len(x[:, 0]), 1)), x), axis=1) @ self.theta
        p = 1 / (1 + np.exp(-score))
        prediction = np.where(p >= 0.5, 1., 0.)
        correct = np.where(prediction==y, 1, 0)
        accuracy = np.sum(correct) / len(p)
        if print_result:
            print(f"prediction = {prediction.flatten()}")
            print(f"accuracy = {accuracy}")
            print(f"theta = {self.theta.flatten()}")

        return prediction, accuracy

model = linear_model()
model.load_training_set(x, y)
model.predict(x_test, y_test)
model.iter()
model.print_model()
# model.predict(x_test, y_test)

for i in range(350):
    model.iter()

# model.predict(model.x[:, 1:], model.y)
model.predict(x_test, y_test)
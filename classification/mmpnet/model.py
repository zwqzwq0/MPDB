from preprocess import data_preprocess
import numpy as np
from sklearn import ensemble, linear_model, tree, svm, neighbors
from torch import nn

from sklearn.metrics import *


def create_ml_model_list():
    SVM = svm.SVC()
    KNeighborsClassifier = neighbors.KNeighborsClassifier()
    return [SVM, KNeighborsClassifier]


def try_different_ml_model(models):
    x_train, x_test, y_train, y_test = data_preprocess()
    for model in models:
        # num = models.index(model) + 1
        model.fit(x_train, y_train)
        prediction_result = model.predict(x_test).reshape(-1, 1)
        print(model.__class__.__name__)
        print('accuracy_score: ', accuracy_score(y_test, prediction_result))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.input_stack = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )

        self.output_stack = nn.Sequential(
            nn.Linear(1280, 200),
            nn.ReLU(),
            nn.Linear(200, 5),
        )

    def forward(self, x):
        x_forward = self.input_stack(x)
        x_forward = x_forward.view(x.shape[0], -1)
        output = self.output_stack(x_forward)
        return output


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128000, 20000),
            nn.ReLU(),
            nn.Linear(20000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        x_forward = self.linear_relu_stack(x)
        return x_forward

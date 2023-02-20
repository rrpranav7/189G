from base.base_class.method import method
from base.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Method_CNN(nn.Module):
    max_epoch = 100
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, num_classes, num_channels = 1):
        method.__init__(self, mName, mDescription)
        super(Method_CNN, self).__init__()
        # For grayscale and gray images, the input channels is 1, for RGB it is 3
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*7*7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.output_func = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        y_pred = self.output_func(self.fc2(x))
        return y_pred

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        epoch_list=[]
        accuracy_list=[]
        loss_list = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X)
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%10 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                accuracy_evaluate = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluate['accuracy'],
                'Precision:', accuracy_evaluate['precision'],
                'recall:', accuracy_evaluate['recall'],
                'F1 Score:', accuracy_evaluate['F1 Score'], 'Loss:', train_loss.item())
                epoch_list.append(epoch)
                accuracy_list.append(accuracy_evaluate['accuracy'])
                loss_list.append(train_loss.item())

        plt.plot(epoch_list, accuracy_list)

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.show()

        plt.plot(epoch_list, loss_list)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.show()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
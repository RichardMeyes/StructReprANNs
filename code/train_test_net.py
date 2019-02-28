import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, layers):
        self.layers = layers

        # create Net
        super(Net, self).__init__()

        # build input layer
        self.h0 = nn.Linear(28 * 28, self.layers[0])

        # build hidden layers 1, n-1
        for i_layer in range(len(layers)-1):
            self.__setattr__("h{0}".format(i_layer+1),
                             nn.Linear(self.layers[i_layer], self.layers[i_layer+1]))

        # build output layer
        self.output = nn.Linear(self.layers[-1], 10)

    def forward(self, x):
        for i_layer in range(len(self.layers)):
            x = F.relu(self.__getattr__("h{0}".format(i_layer))(x))
        x = F.log_softmax(self.output(x), dim=1)  # needs NLLLos() loss
        return x

    def train_net(self, criterion, optimizer, trainloader, epochs, device):
        # save untrained net
        torch.save(net.state_dict(), "../nets/MNIST_MLP_{0}_untrained.pt".format(self.layers))

        # train the net
        log_interval = 10
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                data = data.view(-1, 28 * 28)
                optimizer.zero_grad()
                net_out = self(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                                   len(trainloader.dataset),
                                                                                   100. * batch_idx / len(trainloader),
                                                                                   loss.data.item()))
        # save trained net
        torch.save(net.state_dict(), "../nets/MNIST_MLP_{0}_trained.pt".format(self.layers))

    def test_net(self, criterion, testloader, device):
        # test the net
        test_loss = 0
        correct = 0
        correct_class = np.zeros(10)
        correct_labels = np.array([], dtype=int)
        class_labels = np.array([], dtype=int)
        for i_batch, (data, target) in enumerate(testloader):
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            batch_labels = pred.eq(target.data)
            correct_labels = np.append(correct_labels, batch_labels)
            class_labels = np.append(class_labels, target.data)
            for i_label in range(len(target)):
                label = target[i_label].item()
                correct_class[label] += batch_labels[i_label].item()
            correct += batch_labels.sum()
        test_loss /= len(testloader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(test_loss, correct,
                                                                                     len(testloader.dataset),
                                                                                     100. * correct.item() / len(
                                                                                         testloader.dataset)))
        acc = 100. * correct.item() / len(testloader.dataset)
        # calculate class_acc
        acc_class = np.zeros(10)
        for i_label in range(10):
            num = (testloader.dataset.test_labels.numpy() == i_label).sum()
            acc_class[i_label] = correct_class[i_label]/num
        return acc, correct_labels, acc_class, class_labels


if __name__ == "__main__":

    # setting flags
    flag_train = False
    flag_test = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build net
    layer_structure = (100, 100, 100)
    net = Net(layers=layer_structure)
    net.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    if flag_train:
        net.train_net(criterion, optimizer, trainloader, epochs=100, device=device)
    else:
        net.load_state_dict(torch.load("../nets/MNIST_MLP_{0}_trained.pt".format(layer_structure)))
        net.eval()

    if flag_test:
        acc, correct_labels, acc_class, _ = net.test_net(criterion, testloader, device)
        print(acc)
        print(correct_labels)
        print(acc_class)
        print(acc_class.mean())  # NOTE: This does not equal to the calculated total accuracy as the distribution of labels is not equal in the test set!
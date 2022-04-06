import time
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from operator import itemgetter
import random
import torch.nn.utils.prune as prune


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


def load_data(batch_size):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = CIFAR10("data", train=False, download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size)
    num_examples = {"testset": len(testset)}

    return  testloader, num_examples

def get_cifar_iid(batch_size, total_num_clients, id):



    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='data', train=True,download=False, transform=transform) # todo
    total_data_train = len(trainset)
    random_list_train = random.sample(range(total_data_train), total_data_train)
    data_per_client_train = int(total_data_train / total_num_clients)
    indexes_train = random_list_train[id*data_per_client_train: (id+1)*data_per_client_train]
    trainset = (list(itemgetter(*indexes_train)(trainset)))
    trainloader = (torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True))

    testset = CIFAR10(root='data', train=False,download=False, transform=transform)
    testloader = (torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False))

    return trainloader, testloader

def test_server(net, testloader, device):

    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

class CustomMask(prune.BasePruningMethod):

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

def foobar_unstructured(module, name):

    CustomMask.apply(module, name)
    return module
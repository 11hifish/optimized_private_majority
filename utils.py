from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torchvision.transforms as transforms


# load MNIST dataset
def load_MNIST_full():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    print(type(train_data), train_data.data.size())

    return train_data, test_data

def load_SVHN_full():
    train_data = datasets.SVHN(
        root='data',
        split='train',
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.SVHN(
        root='data',
        split='test',
        transform=ToTensor(),
        download=True
    )
    # print(type(train_data), train_data.data.size())
    print(type(train_data))
    print(len(train_data.data))

    return train_data, test_data

def load_CIFAR10_full():
    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                 download=True, transform=transform)
    print(type(train_data))
    print(len(train_data.data))
    return train_data, test_data


def load_MNIST_from_target_class():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    print(type(train_data), train_data.data.size())

    # get data from target classes
    idx = (train_data.targets == 5) | (train_data.targets == 8)
    train_data.data = train_data.data[idx]
    train_data.targets = train_data.targets[idx]
    train_data.targets[train_data.targets == 5] = 0
    train_data.targets[train_data.targets == 8] = 1
    print(train_data.data.size(), train_data.targets.size())  # torch.Size([11272, 28, 28]) torch.Size([11272])

    idx_test = (test_data.targets == 5) | (test_data.targets == 8)
    test_data.data = test_data.data[idx_test]
    test_data.targets = test_data.targets[idx_test]
    test_data.targets[test_data.targets == 5] = 0
    test_data.targets[test_data.targets == 8] = 1
    print(test_data.data.size(), test_data.targets.size())  # torch.Size([1866, 28, 28]) torch.Size([1866])

    return train_data, test_data


def split_data(dataset, K):
    # split dataset into K dataset of equagitl size
    teacher_datasets = []
    batch_size = len(dataset) // K
    print('# samples per worker: ', batch_size)
    start_idx = 0
    for i in range(K):
        end_idx = start_idx + batch_size
        indices = range(start_idx, end_idx)
        teacher_data = torch.utils.data.Subset(dataset, indices)
        start_idx = end_idx
        teacher_datasets.append(teacher_data)
    return teacher_datasets
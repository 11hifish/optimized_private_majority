import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.net_mnist import CNN
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
from opacus import PrivacyEngine
import os


device = 'cuda:5'

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

print(type(train_data), train_data.data.size())


# get data from target classes
idx = (train_data.targets == 5) | (train_data.targets == 8)
train_data.data = train_data.data[idx]
train_data.targets = train_data.targets[idx]
print(train_data.data.size(), train_data.targets.size())  # torch.Size([11272, 28, 28]) torch.Size([11272])

idx_test = (test_data.targets == 5) | (test_data.targets == 8)
test_data.data = test_data.data[idx_test]
test_data.targets = test_data.targets[idx_test]
print(test_data.data.size(), test_data.targets.size())  # torch.Size([1866, 28, 28]) torch.Size([1866])

# now split data
K = 11  # number of teachers
data_split_indices = []
batch_size = len(train_data.data) // K
start_idx, end_idx = 0, 0
for i in range(K - 1):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    data_split_indices.append((start_idx, end_idx))
data_split_indices.append((end_idx, len(train_data.data)))


def train_teacher(train_data_teacher, teacher_idx=0, num_epochs=10, save_folder=None):
    if save_folder is not None and not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    # privacy parameters
    sigma = 10
    max_per_sample_grad_norm = 1
    delta = 0.001  # final eps for 10 epochs = 0.1075
    # delta = 1e-8  # final eps for 10 epochs = 0.2825
    if save_folder is not None and not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    print('len of train data teacher: ', len(train_data_teacher))
    train_dataloader = torch.utils.data.DataLoader(train_data_teacher,
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=1)


    model = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # privacy engine
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
    )

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # gives batch data, normalize x when iterate train_loader
            b_x = images.to(device)  # batch x
            b_y = labels.to(device)  # batch y

            output = model(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
        epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
        print('epoch {}, epsilon: {:.4f}, delta: {:.4f}'.format(epoch, epsilon, delta))
    if save_folder is not None:
        save_path = os.path.join(save_folder, 'teacher_{}.model'.format(teacher_idx))
        torch.save(model._module.state_dict(), save_path)
    return model


def test_teacher_accuracy(teacher_model, test_data):
    teacher_model.eval()
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=1)
    with torch.no_grad():
        correct = 0
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            # gives batch data, normalize x when iterate train_loader
            images = images.to(device)  # batch x
            labels = labels.to(device)  # batch y
            test_output, last_layer = teacher_model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            n_corr = (pred_y == labels).sum().item()
            correct += n_corr
    acc = correct / len(test_data.data)
    print('# test data: {}, acc: {:.4f}'.format(len(test_data.data), acc))

for teacher_idx in range(K):
    start_idx, end_idx = data_split_indices[teacher_idx]
    indices = range(start_idx, end_idx)
    train_teacher_data = torch.utils.data.Subset(train_data, indices)
    teacher_model = train_teacher(train_teacher_data, teacher_idx, save_folder='private_mnist_teachers')
    test_teacher_accuracy(teacher_model, test_data)

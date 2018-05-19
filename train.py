import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import matplotlib.pyplot as plt
import math

from model import SparseAutoencoder

# global constants
BATCH_SIZE = 32
BETA = 3
RHO = 0.01
N_INP = 784
N_HIDDEN = 300
N_EPOCHS = 1
use_sparse = False

rho = torch.FloatTensor([RHO for _ in range(N_HIDDEN)]).unsqueeze(0)

# FashionMNIST data loading
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor()])
train_set = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=False)

auto_encoder = SparseAutoencoder(N_INP, N_HIDDEN)
optimizer = optim.Adam(auto_encoder.parameters(), lr=1e-3)

def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2

# set plot and view data for visualization
N_COLS = 8
N_ROWS = 4
view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]
plt.figure(figsize=(20, 4))

for epoch in range(N_EPOCHS):
    for b_index, (x, _) in enumerate(train_loader):
        x = x.view(x.size()[0], -1)
        x = Variable(x)
        encoded, decoded = auto_encoder(x)
        MSE_loss = (x - decoded) ** 2
        MSE_loss = MSE_loss.view(1, -1).sum(1) / BATCH_SIZE
        if use_sparse:
            rho_hat = torch.sum(encoded, dim=0, keepdim=True)
            sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
            loss = MSE_loss + sparsity_penalty
        else:
            loss = MSE_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: [%3d], Loss: %.4f" %(epoch + 1, loss.data))

for i in range(N_ROWS * N_COLS):
    # original image
    r = i // N_COLS
    c = i % N_COLS + 1
    ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)
    plt.imshow(view_data[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed image
    ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c + N_COLS)
    x = Variable(view_data[i])
    e, y = auto_encoder(x.view(1, -1))
    plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

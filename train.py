import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import matplotlib.pyplot as plt

from model import SparseAutoencoder

# global constants
BATCH_SIZE = 32
BETA = 5
RHO = 0.05
N_INP = 784
N_HIDDEN = 1000
N_EPOCHS = 25
LOG_STEPS = 100
N_VIEW = 10
use_sparse = True

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

# set plot and view data for visualization
plt.figure(figsize=(20, 4))
view_data = [test_set[i][0] for i in range(N_VIEW)]

for epoch in range(N_EPOCHS):
    # log info
    print("Starting epoch: [%3d]" %(epoch + 1))

    for b_index, (x, _) in enumerate(train_loader):
        x = x.view(x.size()[0], -1)
        x = Variable(x)
        encoded, decoded = auto_encoder(x)
        MSE_loss = (x - decoded) ** 2
        MSE_loss = MSE_loss.view(1, -1).sum(1)
        if use_sparse:
            rho_hat = torch.sum(encoded, dim=0, keepdim=True)
            sparsity_penalty = BETA * F.kl_div(rho_hat, rho)
            loss = MSE_loss + sparsity_penalty
        else:
            loss = MSE_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (b_index + 1) % LOG_STEPS == 0:
            print("\tLoss: %.4f" %(loss.data))

    for i in range(N_VIEW):
        # original image
        ax = plt.subplot(2, N_VIEW, i + 1)
        plt.imshow(view_data[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed image
        ax = plt.subplot(2, N_VIEW, i + 1 + N_VIEW)
        plt.subplot
        x = Variable(view_data[i])
        e, y = auto_encoder(x.view(1, -1))
        print(e.sum())
        plt.imshow(y.detach().squeeze().numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


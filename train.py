import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import matplotlib.pyplot as plt

from model import SparseAutoencoder

# global constants
BATCH_SIZE = 32
BETA = 0.1
RHO = 0.05
N_INP = 784
N_HIDDEN = 1000
N_EPOCHS = 5

rho = torch.FloatTensor([RHO for _ in range(N_INP)]).unsqueeze(0)

# MNIST data loading
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

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

for epoch in range(N_EPOCHS):
    for b_index, (x, _) in enumerate(train_loader):
        x = x.view(x.size()[0], -1)
        x = Variable(x)
        target = x
        encoded, decoded = auto_encoder(x)
        rho_hat = torch.sum(decoded, dim=0, keepdim=True)
        MSE_loss = (target - decoded) ** 2
        sparsity_penalty = BETA * F.kl_div(rho_hat, rho)
        print(MSE_loss.size())
        print(sparsity_penalty.size())
        loss = MSE_loss + sparsity_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

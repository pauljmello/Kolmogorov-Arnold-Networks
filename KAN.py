import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class KANLayer(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu'):
        super(KANLayer, self).__init__()
        self.size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.device = device

        self.grid = torch.einsum('i,j->ij', torch.ones(self.size, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid, requires_grad=False)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, (torch.rand(self.size, self.grid.shape[1]) - 0.5) * noise_scale / num, self.grid, k, device))
        self.scale_base = torch.nn.Parameter(torch.ones(self.size, device=device) * scale_base, requires_grad=sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(self.size, device=device) * scale_sp, requires_grad=sp_trainable)
        self.base_fun = base_fun
        self.mask = torch.nn.Parameter(torch.ones(self.size, device=device), requires_grad=False)
        self.grid_eps = grid_eps

    def forward(self, x):
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        preacts = x.permute(1, 0).reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k, device=self.device).permute(1, 0)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)
        return y, preacts, postacts

class KAN(nn.Module):
    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, base_fun=torch.nn.SiLU(), device='cpu'):
        super(KAN, self).__init__()
        self.depth = len(width) - 1
        self.width = width
        self.act_fun = nn.ModuleList()
        self.device = device

        for l in range(self.depth):
            scale_base = 1 / np.sqrt(width[l]) + (torch.randn(width[l] * width[l + 1], ) * 2 - 1) * noise_scale
            self.act_fun.append(KANLayer(in_dim=width[l], out_dim=width[l + 1], num=grid, k=k, noise_scale=noise_scale, scale_base=scale_base, base_fun=base_fun, device=device))

    def forward(self, x):
        for l in range(self.depth):
            x, _, _ = self.act_fun[l](x)
        return x

def coef2curve(x_eval, grid, coef, k, device="cpu"):
    return torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    def extend_grid(grid, k_extend):
        h = (grid[:, -1] - grid[:, 0]) / (grid.size(1) - 1)
        left_ext = grid[:, 0].unsqueeze(1) - h.unsqueeze(1) * torch.arange(k_extend, 0, -1, device=device).unsqueeze(0)
        right_ext = grid[:, -1].unsqueeze(1) + h.unsqueeze(1) * torch.arange(1, k_extend + 1, device=device).unsqueeze(0)
        return torch.cat([left_ext, grid, right_ext], dim=1)

    if extend:
        grid = extend_grid(grid, k)
    x = x.unsqueeze(1)
    grid = grid.unsqueeze(2)

    if k == 0:
        return ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
    else:
        B_km1 = B_batch(x[:, 0], grid[:, :, 0], k - 1, extend=False, device=device)
        left_term = (x - grid[:, :-k - 1]) / (grid[:, k:-1] - grid[:, :-k - 1]) * B_km1[:, :-1]
        right_term = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * B_km1[:, 1:]
        return left_term + right_term

def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    return torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]

def train_kan(model, train_loader, criterion, optimizer, num_epochs=100, device='cpu', lambda1=1e-3, lambda2=1e-3):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = sparsification_regularization(model, lambda1, lambda2)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}')

def sparsification_regularization(model, lambda1=1e-3, lambda2=1e-3):
    l1_loss = 0
    entropy_loss = 0
    for layer in model.act_fun:
        for coeff in layer.coef:
            l1_loss += torch.sum(torch.abs(coeff))
            prob = torch.softmax(coeff, dim=0)
            entropy_loss += -torch.sum(prob * torch.log(prob + 1e-6))
    return lambda1 * l1_loss + lambda2 * entropy_loss

def test_kan(model, test_loader, criterion, device='cpu'):
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    # Generate Data
    x = torch.linspace(0, 5, 250).unsqueeze(1)
    y = torch.sin(2 * torch.pi * x)

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    spline_order = 10
    model = KAN(width=[1, 5, 5, 1], grid=5, k=spline_order, device='cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_kan(model, train_loader, criterion, optimizer, num_epochs=100, device='cpu', lambda1=1e-3, lambda2=1e-3)

    test_kan(model, train_loader, criterion, device='cpu')

    # Plot the output
    with torch.no_grad():
        output = model(x).cpu()
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='True Function')
    plt.plot(x.cpu().numpy(), output.numpy(), label='KAN Output')
    plt.legend()
    plt.show()

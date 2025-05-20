import torch
from torch import nn


class KAN:

    class Layer(nn.Module):
        """Kolmogorov-Arnold Network Layer with B-splines

        Implementation of the Kolmogorov-Arnold representation theorem:
        f(x₁,...,xₙ) = ∑ᵢ₌₁ᵐ Φᵢ(∑ⱼ₌₁ⁿ φᵢⱼ(xⱼ))

        Using B-splines for universal function approximation:
        φᵢⱼ(x) = ∑ₖ cᵏᵢⱼ Bᵏ(x)
        where Bᵏ are B-spline basis functions of degree k
        """

        def __init__(self, in_dim, out_dim, num, k, device):
            super().__init__()
            self.size = out_dim * in_dim
            self.out_dim = out_dim
            self.in_dim = in_dim
            self.num = num                # Number of control points in B-spline: determines grid resolution
            self.k = k                    # Degree of B-spline: determines smoothness
            self.device = device

            # Initialize B-spline grid
            self.grid = nn.Parameter(torch.einsum( 'i,j->ij', torch.ones(self.size, device=device), torch.linspace(-1, 1, steps=num + 1, device=device)), requires_grad=False)

            # Initialize coefficients
            noise_scale = 0.1 / num
            init_noise = (torch.rand(self.size, self.grid.shape[1], device=device) - 0.5) * noise_scale
            self.coef = nn.Parameter(BSplineUtils.curve2coef(self.grid, init_noise, self.grid, k, device))
            self.scale_base = nn.Parameter(torch.ones(self.size, device=device))
            self.scale_sp = nn.Parameter(torch.ones(self.size, device=device))
            self.base_fun = nn.SiLU()

        def forward(self, x):
            x = x.to(self.device)
            batch = x.shape[0]

            # Expand dimensions
            x_expanded = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device))
            x_expanded = x_expanded.reshape(batch, self.size).permute(1, 0)

            # Apply base function (SiLU) and B-spline transformations
            base = self.base_fun(x_expanded).permute(1, 0)  # Base function: SiLU(xⱼ)
            y = BSplineUtils.coef2curve(x_expanded, self.grid, self.coef, self.k, self.device).permute(1, 0)  # B-spline: φᵢⱼ(xⱼ)

            # Combine base and spline components with learned scales
            y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y

            # Sum over input dimensions to get output: yᵢ = ∑ⱼ₌₁ⁿ [...]
            y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)
            return y

    class Network(nn.Module):
        """Kolmogorov-Arnold Network

        Implements a multi-layer KAN network composed of KAN layers:
        f(x) = f_L ∘ f_{L-1} ∘ ... ∘ f_1(x)

        where each f_l is a KAN layer implementing the Kolmogorov-Arnold representation
        """

        def __init__(self, width, grid, k, device):
            super().__init__()
            self.layers = nn.ModuleList()
            self.device = device
            self.train_losses = []
            self.val_losses = []

            for i in range(len(width) - 1):
                self.layers.append(KAN.Layer(width[i], width[i + 1], grid, k, device))

        def forward(self, x):
            x = x.to(self.device)
            for layer in self.layers:
                x = layer(x)
            return x

    @staticmethod
    def reshape_sequence(inputs, targets):
        batch_size, seq_len, feat_dim = inputs.shape
        inputs_reshaped = inputs.reshape(batch_size * seq_len, feat_dim)
        targets_reshaped = targets.reshape(batch_size * seq_len, feat_dim)
        return inputs_reshaped, targets_reshaped

    @staticmethod
    def train(model, train_loader, val_loader, epochs, learning_rate, print_every, device, reg_lambda=1e-4):
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Reshape for KAN
                inputs_reshaped, targets_reshaped = KAN.reshape_sequence(inputs, targets)

                optimizer.zero_grad()
                outputs = model(inputs_reshaped)
                loss = criterion(outputs, targets_reshaped)

                l1_loss = sum(torch.sum(torch.abs(layer.coef)) for layer in model.layers)  # ∑ᵢ|cᵢ| - sum of absolute values of all coefficients
                loss += reg_lambda * l1_loss

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    inputs_reshaped, targets_reshaped = KAN.reshape_sequence(inputs, targets)
                    outputs = model(inputs_reshaped)
                    loss = criterion(outputs, targets_reshaped)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}')

        model.train_losses = train_losses
        model.val_losses = val_losses

        return model, {'train': train_losses, 'val': val_losses}

    @staticmethod
    def create(width, grid, k, device):
        return KAN.Network(width, grid, k, device)


class BSplineUtils:

    @staticmethod
    def extend_grid(grid, k_extend, device):
        """Extend the B-spline grid to handle boundary conditions.

        Adds k_extend knots on each side of the grid with uniform spacing:
        t_{-k},...,t_{-1}, t_0,...,t_n, t_{n+1},...,t_{n+k}
        """
        h = (grid[:, -1] - grid[:, 0]) / (grid.size(1) - 1)
        left_ext = grid[:, 0].unsqueeze(1) - h.unsqueeze(1) * torch.arange(k_extend, 0, -1, device=device).unsqueeze(0)
        right_ext = grid[:, -1].unsqueeze(1) + h.unsqueeze(1) * torch.arange(1, k_extend + 1, device=device).unsqueeze(0)
        return torch.cat([left_ext, grid, right_ext], dim=1)

    @staticmethod
    def B_batch(x, grid, k, extend, device):
        """Calculate batched B-spline basis functions"""
        x = x.to(device)
        grid = grid.to(device)

        if extend:
            grid = BSplineUtils.extend_grid(grid, k, device)
        x = x.unsqueeze(1)
        grid = grid.unsqueeze(2)

        if k == 0:
            # B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, 0 otherwise
            return ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
        else:
            # Recursive formula for higher-degree basis functions
            B_km1 = BSplineUtils.B_batch(x[:, 0], grid[:, :, 0], k - 1, False, device)

            # Left term: (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
            left_term = (x - grid[:, :-k - 1]) / (grid[:, k:-1] - grid[:, :-k - 1] + 1e-10) * B_km1[:, :-1]

            # Right term: (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            right_term = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-10) * B_km1[:, 1:]

            return left_term + right_term

    @staticmethod
    def coef2curve(x_eval, grid, coef, k, device):
        x_eval = x_eval.to(device)
        grid = grid.to(device)
        coef = coef.to(device)

        # Convert coefficients to curve values using B-splines
        return torch.einsum('ij,ijk->ik', coef, BSplineUtils.B_batch(x_eval, grid, k, True, device))

    @staticmethod
    def curve2coef(x_eval, y_eval, grid, k, device):
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        grid = grid.to(device)

        # Compute the basis function matrix B at x_eval points
        mat = BSplineUtils.B_batch(x_eval, grid, k, True, device).permute(0, 2, 1)

        # Solve the linear system B·c = y for coefficients c using least squares
        return torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
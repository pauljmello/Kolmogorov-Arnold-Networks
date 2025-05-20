import torch
import numpy as np
from kan import KAN
from utils import prepare_data, visualize_results, ensure_output_dir


def main():
    seed = 1001
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model hyperparameters
    input_dim = 1
    hidden_dim = 32
    output_dim = 1
    width = [input_dim, hidden_dim, output_dim]  # [in_dim, <-hidden layers->, out_dim]
    grid = 8                # Number of control points in B-spline
    k = 3                   # Degree of B-spline

    # Training hyperparameters
    batch_size = 512
    epochs = 250
    learning_rate = 0.003
    print_every = 10
    save_dir = 'images'
    reg_lambda = 1e-4

    # Dataset hyperparameters
    num_samples = 10000
    seq_length = 100
    max_time = 10.0
    train_split = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ensure_output_dir(save_dir)

    print("Preparing data...")
    train_loader, val_loader, vis_sample = prepare_data(num_samples, seq_length, batch_size, max_time, train_split, seed)

    print("Creating model...")
    model = KAN.create(width, grid, k, device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    print("Training model...")
    model, losses = KAN.train(model, train_loader, val_loader, epochs, learning_rate, print_every, device, reg_lambda)

    print("Visualizing results...")
    visualize_results(model, vis_sample, save_dir, device)


if __name__ == "__main__":
    main()
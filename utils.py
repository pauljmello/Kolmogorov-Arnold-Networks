import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from kan import KAN


def prepare_data(num_samples, seq_length, batch_size, max_time, train_split, seed):
    time = np.linspace(0, max_time, num_samples)
    signals = np.sin(time) * np.exp(-0.1 * time)  # Damped sine wave

    X, y = [], []
    for i in range(len(signals) - seq_length):
        X.append(signals[i:i + seq_length])
        y.append(signals[i + 1:i + seq_length + 1])

    X = np.array(X)[:, :, np.newaxis]  # Add feature dimension
    y = np.array(y)[:, :, np.newaxis]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    indices = np.arange(len(X_tensor))
    train_idx, val_idx = train_test_split(indices, test_size=(1 - train_split), random_state=seed)

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, prefetch_factor=2, persistent_workers=True)

    return train_loader, val_loader, (X_val[:3], y_val[:3])  # Return sample for visualization


def visualize_results(model, sample_data, save_dir, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)

    X_sample, y_true = sample_data
    X_sample = X_sample.to(device)

    model.eval()
    with torch.no_grad():
        # For KAN, we need to reshape the sequence data
        batch_size, seq_len, feat_dim = X_sample.shape
        X_reshaped, _ = KAN.reshape_sequence(X_sample, X_sample)  # Using KAN's reshape method

        # Get predictions
        y_pred_reshaped = model(X_reshaped)

        # Reshape back to sequence format
        y_pred = y_pred_reshaped.reshape(batch_size, seq_len, feat_dim)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    feature_dim = y_true.shape[2]

    # Plot and save prediction comparisons for each sample
    for i in range(len(y_true)):
        fig, axes = plt.subplots(feature_dim, 1, figsize=(10, 3 * feature_dim))
        if feature_dim == 1:
            axes = [axes]

        for j in range(feature_dim):
            axes[j].plot(y_true[i, :, j], 'b-', label='True')
            axes[j].plot(y_pred[i, :, j], 'r--', label='Predicted')
            axes[j].set_title(f'Sample {i + 1}, Feature {j + 1}')
            axes[j].legend()
            axes[j].grid(True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{i + 1}_prediction.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved prediction plot to {save_path}")

    # Plot and save training/validation loss curves
    plt.figure(figsize=(10, 4))
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    loss_plot_path = os.path.join(save_dir, 'training_validation_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}")


def ensure_output_dir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
# Kolmogorov-Arnold Networks (KAN)

Kolmogorov-Arnold Networks (KANs) are a novel type of neural network that replace traditional fixed weights with learnable functions (splines) on the edges. This approach offers improved accuracy and interpretability over traditional models like MLPs, particularly for small-scale tasks.

## Key Features

- **Learnable Functions**: Replaces fixed weights with flexible, learnable functions.
- **Spline-Based**: Utilizes spline functions for better accuracy in low-dimensional tasks.
- **Dynamic Grids**: Allows for grid refinement during training to improve model accuracy.

## Getting Started

### Prerequisites

Install the necessary Python packages:

```bash
pip install torch matplotlib numpy
```

### Running the Code

1. **Generate Data**: The script includes a simple dataset generator, such as a sine wave.

2. **Build the Model**:

    ```python
    model = KAN(width=[1, 5, 5, 1], grid=5, k=10, device='cpu')
    ```

3. **Train the Model**:

    ```python
    train_kan(model, train_loader, criterion, optimizer, num_epochs=100, device='cpu')
    ```

4. **Test and Visualize**:

    ```python
    test_kan(model, test_loader, criterion, device='cpu')
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='True Function')
    plt.plot(x.cpu().numpy(), output.numpy(), label='KAN Output')
    plt.legend()
    plt.show()
    ```

## Advanced Features

- **Sparsification**: Automatically reduces network size for better interpretability.
- **Grid Extension**: Refines the function resolution during training for improved accuracy.

---

Explore the code, tweak the parameters, and discover how KANs can improve your tasks!


## Citation

If you use this code in your research, please cite:

```bibtex
@software{pjmKAN2024,
  author = {Paul J Mello},
  title = {Kolmogorov-Arnold Networks},
  url = {https://github.com/pauljmello/Kolmogorov-Arnold-Networks},
  year = {2024},
}

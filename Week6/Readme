# MNIST Denoising Autoencoder

A deep learning project that removes noise from handwritten digit images using autoencoders, trained and evaluated on the MNIST dataset.

## What this does

1. Loads and preprocesses the MNIST dataset.
2. Adds artificial Gaussian noise to create noisy versions of the images.
3. Trains three different autoencoder architectures to reconstruct clean images from the noisy inputs:
   - **FFNN Autoencoder** — simple linear encoder/decoder.
   - **Transpose CNN Autoencoder** — convolutional encoder, transposed-convolution decoder.
   - **Upsampled CNN Autoencoder** — convolutional encoder, nearest-neighbor upsampling + convolution decoder.
4. Evaluates each model on the test set and visualizes Original / Noisy / Denoised images side by side.

## Requirements

- Python 3.8+
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

Install with:
```bash
pip install torch torchvision numpy matplotlib
```

## How to run

1. Open `autoencoder_mnist.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Run all cells top to bottom.
3. The MNIST dataset is downloaded automatically on first run — no manual setup needed.
4. Trained model weights (`.pth` files) are saved automatically during training.
5. Training all three models for 20 epochs each takes a few minutes on CPU, faster on GPU.

No files need to be downloaded or prepared beforehand — the notebook is self-contained.

## Results

See the **Observations & Analysis** section at the end of the notebook for a full write-up, including a comparison of validation loss across all three models and notes on challenges encountered during training.

## Files

| File | Description |
|---|---|
| `autoencoder_mnist.ipynb` | Main notebook — all code, training, and results. |
| `data/` *(auto-generated)* | MNIST dataset, downloaded automatically. |
| `*.pth` *(auto-generated)* | Saved model weights after training. |

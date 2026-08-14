# CS 410 – Assignment 4: Convolutional Autoencoders on Fashion-MNIST

Programming assignment for CS 4/510: Computer Vision & Deep Learning, PSU. Builds and trains convolutional autoencoders (CAEs) on the Fashion-MNIST dataset using Keras/TensorFlow, covering reconstruction, denoising, and latent-space visualization.

## What it does

**Part 1 — Build the model.** A convolutional autoencoder that compresses a 28x28 grayscale image down to a 2-number latent code, then reconstructs it back to 28x28 through an encoder/decoder pair (Conv2D + MaxPooling down, Conv2D + UpSampling back up).

**Part 2 — Train & reconstruct.** Trains the autoencoder to reproduce its own input (binary cross-entropy loss, Adam optimizer, 15 epochs). Plots train/test loss over time and saves side-by-side original vs. reconstructed image examples.

**Part 3 — Denoising autoencoder.** Adds Gaussian noise to the inputs and trains a second, freshly-initialized autoencoder (same architecture) to map noisy images back to their clean originals. Saves loss curves and original/noisy/denoised examples.

**Part 4 — Latent space visualization.** Runs a random sample of 1,000 test images through the trained encoder to get their 2D codes, and plots them colored by class label. Does the same with PCA (2 components) on the flattened raw images for comparison.

## Files

- `CS410_Computer_Vision_and _Deep_Learning_assignment_4.py` — main script
- `CS410 computer vision and deep learning assignment 4.pdf` — writeup
- `step2_loss.png`, `step2_examples.png` — CAE training loss + reconstructions
- `step3_loss.png`, `step3_examples.png` — denoising CAE loss + examples
- `step4_cae_latent.png` — 2D latent space colored by class
- `step4_pca.png` — PCA 2D projection colored by class, for comparison

## Requirements

```
numpy
tensorflow
keras
matplotlib
scikit-learn
```

Fashion-MNIST is downloaded automatically via `tf.keras.datasets.fashion_mnist.load_data()` on first run.

## Usage

```bash
python "CS410_Computer_Vision_and _Deep_Learning_assignment_4.py"
```

This trains both autoencoders end-to-end (30 epochs total across the two models) and saves all six output images to the working directory. Training uses a GPU if available but will run on CPU.

## Notes

- Latent dimensionality is fixed at 2 so the codes can be plotted directly without further dimensionality reduction.
- The denoising autoencoder is trained from scratch rather than fine-tuned from Part 2's weights, to isolate the effect of the noisy-input task.

"""
Veniamin Velikoretskikh
veniamin@pdx.edu
CS 4/510: Computer Vision & Deep Learning
Programming Assignment #4
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.decomposition import PCA

# fashion-mnist class names, just used for labeling the plots later
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

EPOCHS = 15
BATCH_SIZE = 128




# load the data. already set to  (60000, 28, 28) with pixel values 0-255
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize pixel values to 0-1 and add a channel dimension so conv2d can use it
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

print("train shape:", x_train.shape)
print("test shape:", x_test.shape)


#Part 1: build the convolutional autoencoder model. the encoder compresses a 28x28 image down to just 2 numbers,
# and the decoder expands those 2 numbers back up to a 28x28 image
 
def build_autoencoder():
    #  encoder: take a 28x28 image and compress it down to just 2 numbers 
    encoder_input = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)     # now 7x7x8

    # flatten to a 1D vector so we can use a Dense layer to get down to 2 numbers
    x = layers.Flatten()(x)
    code = layers.Dense(2, activation="relu", name="code")(x)

    # decoder: take the 2D code and expand it back up to a 28x28 image
    x = layers.Dense(7 * 7 * 8, activation="relu")(code)
    x = layers.Reshape((7, 7, 8))(x)  # back to a 3D shape so conv2d can use it
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)    # 7 -> 14
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)     # 14 -> 28
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)


    # final layer: 1 channel, 28x28, pixel values 0-1
    decoder_output = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # build the Keras models for the autoencoder and the encoder
    autoencoder = models.Model(encoder_input, decoder_output, name="autoencoder")
    encoder = models.Model(encoder_input, code, name="encoder") 
    return autoencoder, encoder


autoencoder, encoder = build_autoencoder()
autoencoder.summary()


#Part 2: train the autoencoder to just copy its input to its output, which is
# what teaches it to compress the input down to a 2D code and then expand it
# back up to a 28x28 image

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


history = autoencoder.fit(
    x_train, x_train,
    validation_data=(x_test, x_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# plot how loss changed over training
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="test loss")
plt.xlabel("epoch")
plt.ylabel("binary cross-entropy loss")
plt.title("CAE Training/Test Loss")
plt.legend()
plt.savefig("step2_loss.png", dpi=150)
print("Saved: step2_loss.png")
plt.close()

# show some test images next to their reconstructions
reconstructed = autoencoder.predict(x_test[:10])

# plot the original and reconstructed images side by side
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(x_test[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_title("original", loc="left")
axes[1, 0].set_title("reconstructed", loc="left")
plt.savefig("step2_examples.png", dpi=150)
print("Saved: step2_examples.png")
plt.close()



# part 3: denoising autoencoder

# add random Gaussian noise to the images, then train a fresh autoencoder to take 
# the noisy images as input and produce the original clean images as output
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(0, 1, x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(0, 1, x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0).astype("float32")
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0).astype("float32")

# fresh model, same architecture, trained from scratch on the noisy task
denoising_autoencoder, denoising_encoder = build_autoencoder()
denoising_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# train the denoising autoencoder

history_denoise = denoising_autoencoder.fit(
    x_train_noisy, x_train,
    validation_data=(x_test_noisy, x_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# plot how loss changed over training
plt.figure()
plt.plot(history_denoise.history["loss"], label="train loss")
plt.plot(history_denoise.history["val_loss"], label="test loss")
plt.xlabel("epoch")
plt.ylabel("binary cross-entropy loss")
plt.title("Denoising CAE Training/Test Loss")
plt.legend()
plt.savefig("step3_loss.png", dpi=150)
print("Saved: step3_loss.png")
plt.close()

# show some test images next to their noisy versions and the denoised reconstructions
denoised = denoising_autoencoder.predict(x_test_noisy[:10])

fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
for i in range(10):
    axes[0, i].imshow(x_test[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_test_noisy[i].squeeze(), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(denoised[i].squeeze(), cmap="gray")
    axes[2, i].axis("off")
axes[0, 0].set_title("original", loc="left")
axes[1, 0].set_title("noisy", loc="left")
axes[2, 0].set_title("denoised", loc="left")
plt.savefig("step3_examples.png", dpi=150)
print("Saved: step3_examples.png")
plt.close()



#Part 4: visualize the 2D latent space of the encoder by running a random sample of test images through it

np.random.seed(0)
sample_idx = np.random.choice(len(x_test), size=1000, replace=False)
x_sample = x_test[sample_idx]
y_sample = y_test[sample_idx]

# run the encoder on the sample images to get their 2D latent codes
latent_codes = encoder.predict(x_sample)

# plot the 2D latent codes, colored by their class label
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1], c=y_sample, cmap="tab10", s=10)
plt.colorbar(scatter, ticks=range(10), label="class")
plt.xlabel("code dimension 1")
plt.ylabel("code dimension 2")
plt.title("CAE 2D Latent Space")
plt.savefig("step4_cae_latent.png", dpi=150)
print("Saved: step4_cae_latent.png")
plt.close()

# same idea but with PCA instead, on the same images flattened to 1D vectors
x_sample_flat = x_sample.reshape(len(x_sample), -1)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_sample_flat)

# plot the PCA result, colored by their class label
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_sample, cmap="tab10", s=10)
plt.colorbar(scatter, ticks=range(10), label="class")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA 2D Projection")
plt.savefig("step4_pca.png", dpi=150)
print("Saved: step4_pca.png")
plt.close()
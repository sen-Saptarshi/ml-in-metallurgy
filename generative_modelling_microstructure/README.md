# Contrastive Conditional GAN for Microstructure Generation (TensorFlow)

The goal of this project is to generate realistic, deformed microstructures of a High Entropy Alloy ($CoCrFeNiTa_{0.395}$) conditioned on specific processing parameters (temperature and strain rate).

## Model Architecture

This model is a **Conditional Generative Adversarial Network (CGAN)** with several key features from the paper:

1.  **WGAN Framework:** It uses a Wasserstein GAN (WGAN) loss and replaces the discriminator with a **Critic** to improve training stability.
2.  **Conditional Contrastive Loss (ContraGAN):** To combat the paper's problem of sparse data, the Critic is also trained with a **conditional contrastive loss**. This loss "pulls" images with the same label closer together in the representation space and "pushes" images with different labels apart.
3.  **Self-Attention:** The **Generator** includes a **Self-Attention layer** to help capture long-range dependencies and intricate features within the microstructures.
4.  **Deep-Conditional Architecture:** The conditioning labels (processing parameters) are not just concatenated. They are embedded and combined with feature maps deep within both the Generator and the Critic.

## Tech Stack

- Python 3.x
- TensorFlow 2.x (with Keras)
- NumPy
- Matplotlib

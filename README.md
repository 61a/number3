## Enhancing Pulsar Candidate Identification with a Self-Tuning Pseudo-Labeling Semi-Supervised Learning

This repository contains the implementation of a semi-supervised learning system with generative models for the identification of pulsar candidates. With an integrated workflow that leverages Vector Quantized Variational AutoEncoder (VQ-VAE) and Generative Pre-trained Transformer (GPT), our system is designed to effectively generate and identify pulsars from observational data.

## System Overview

- `train_vqvae.py`: This script is used to train the VQ-VAE model on the preprocessed pulsar data.
- `train_GPT.py`: This script is responsible for training the GPT model on the pulsar data representations learned by VQ-VAE.
- `sample_VQVAE_GPT.py`: This script utilizes both trained VQ-VAE and GPT models to generate new pulsar data.
- `Multimodel_Semi.py`: The main executable that integrates the workflow from data preprocessing to pulsar candidate identification.

## Getting Started

### Prerequisites

Before running the scripts, ensure that you have the necessary data preprocessed according to the system's requirements. The following dependencies are also required:

- Python 3.x
- PyTorch
- NumPy

To run the entire workflow, execute the main script Multimodel_Semi.py:
```bash
python Multimodel_Semi.py
```

## License

Distributed under the MIT License. See LICENSE for more information.


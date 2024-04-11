## Enhancing Pulsar Candidate Identification with a Self-Tuning Pseudo-Labeling Semi-Supervised Learning

This repository contains the implementation of a semi-supervised learning system with generative models for the identification of pulsar candidates. With an integrated workflow that leverages Vector Quantized Variational AutoEncoder (VQ-VAE) and Generative Pre-trained Transformer (GPT), our system is designed to effectively generate and identify pulsars from observational data.

## System Overview

- `train_vqvae.py`: This script is used to train the VQ-VAE model on the preprocessed pulsar data.
- `train_GPT.py`: This script is responsible for training the GPT model on the pulsar data representations learned by VQ-VAE.
- `sample_VQVAE_GPT.py`: This script utilizes both trained VQ-VAE and GPT models to generate new pulsar data.
- `Multimodel_Semi.py`: The main executable that integrates the workflow from data preprocessing to pulsar candidate identification.

# EnRoTransformerTranslator - Adapted from Tutorial Implementation

## Overview

This project is a **Transformer-based neural network for English to Romanian machine translation, adapted from a tutorial implementation**.  While the foundational architecture and coding approach were learned from an online tutorial inspired by the "Attention is All You Need" paper, this project **goes beyond a simple code-along by applying these concepts to the English-Romanian language pair using a different dataset.**

The core goal of this project was to **learn and implement the Transformer architecture**, and then **demonstrate the ability to adapt and apply this knowledge to a new machine translation task.**  This involved:

* **Adapting a Transformer Implementation:** Taking a tutorial implementation (originally for English-Italian translation) and modifying it for English-Romanian.
* **Utilizing a Different Dataset:**  Building and using a new English-Romanian dataset (laelhalawani/opus_and_europarl_en_ro), requiring adjustments to data preprocessing and tokenization.
* **Building an End-to-End Translator:** Creating a functional English-Romanian translator, including model training and a Streamlit web application for demonstration.

This project showcases:

* **Neural Machine Translation (NMT) for English-Romanian:**  Translates English text specifically into Romanian.
* **Transformer Architecture (Adapted Implementation):**  Leverages the core components of the Transformer model, adapted from a tutorial, including:
    * **Encoder and Decoder:**  Sequence-to-sequence modeling.
    * **Multi-Head Attention:**  The key mechanism of the Transformer.
    * **Positional Encoding:**  Handling sequence order.
* **Streamlit Web Application:**  Provides an interactive web interface to easily use the English-Romanian translator.
* **Attention Visualization:**  Allows visualization of the attention mechanism in action.

## Demo

This project includes a Streamlit web application that provides a user-friendly interface for translating English text to Romanian using the adapted Transformer model.

![Streamlit App Screenshot](/screenshots/streamlit-app.png)

**Attention Visualization Demo**

![Cross attention visualization](/screenshots/cross-attention-visual.png)

## Architecture and Implementation

This project implements a Transformer-based neural network for English-Romanian translation, **building upon a tutorial implementation originally designed for English-Italian translation.**  While the fundamental architecture and coding structure were learned from this tutorial and are inspired by the "Attention is All You Need" paper, **this project involved significant adaptation and original work to create an English-Romanian translator.**

Key adaptations and components include:

* **Input Embedding:**
    * Uses learned word embeddings to convert input English tokens into dense vector representations.
* **Positional Encoding:**
    * Implements sine and cosine positional encodings to inject information about the position of tokens in the sequence.
* **Layer Normalization:**
    * Applies Layer Normalization within Encoder and Decoder blocks to stabilize training and improve performance.
* **Feed-Forward Block:**
    * Each Encoder and Decoder block includes a Feed-Forward Network, which is a two-layer linear transformation with a ReLU activation in between. This non-linearity is important for model capacity.
* **Multi-Head Attention Block:**
    * Implements Multi-Head Attention, allowing the model to attend to different parts of the input sequence in parallel, capturing richer relationships.
* **Residual Connection:**
    * Employs residual connections (skip connections) around each sub-layer (Multi-Head Attention and Feed-Forward Network) within Encoder and Decoder blocks. This helps with gradient flow and training deeper networks.
* **Encoder Block:**
    * The Encoder is composed of a stack of one Encoder Block. Each Encoder Block consists of a Multi-Head Attention sub-layer followed by a Feed-Forward Network sub-layer, with residual connections and layer normalization around each.
* **Decoder Block:**
    * The Decoder is also composed of a stack of 1 Decoder Block. Each Decoder Block includes Masked Multi-Head Attention, Encoder-Decoder Attention, and a Feed-Forward Network sub-layer, all with residual connections and layer normalization. Masked attention ensures the decoder only attends to previous positions in the output sequence.
* **Projection Layer (Linear Layer):**
    * A final Linear Layer (Projection Layer) is used in the Decoder to project the output of the final Decoder Block to the vocabulary size, allowing the model to predict the probability distribution over Romanian tokens.
* **Transformer (Overall Architecture):**
    * The complete Transformer model is constructed by combining an Input Embedding layer, Positional Encoding, a Encoder Block, a Decoder Block, and a Projection Layer.  It follows the Encoder-Decoder architecture for sequence-to-sequence tasks, driven by the attention mechanism.

* **Dataset Module (for English-Romanian):**  A Python module is included to handle the creation of the bilingual English-Romanian dataset. This **module contains a `BilingualDataset` class and supporting functions** to download and preprocess data from the raw dataset to create training and validation splits suitable for English-Romanian translation.  While the module structure may be inspired by the tutorial, **it is adapted and configured for the specific English-Romanian dataset.**  The raw dataset source is: [laelhalawani/opus_and_europarl_en_ro](https://huggingface.co/datasets/laelhalawani/opus_and_europarl_en_ro).

* **Tokenizer (Specifically Built for English-Romanian):**  A tokenizer was used to convert text into numerical tokens. Crucially, this tokenizer was *built and trained specifically for the English-Romanian dataset used in this project*. This custom-built tokenizer ensures optimal tokenization for the English-Romanian language pair, a key adaptation from the original tutorial which focused on English-Italian.

* **Training Process (Adapted for English-Romanian):**  The model was trained using the **English-Romanian** bilingual dataset created by the provided dataset module and the **custom-built English-Romanian tokenizer**. Training was performed using **Adam** optimizer with a learning rate of **0.0001**.  **CrossEntropyLoss** was used as the loss function. The model was trained for **10** epochs. Training was conducted in a Kaggle Notebook environment, and the notebook (`train_nb.ipynb`) is included in this repository for reference.

**Inspiration and Tutorial Basis:** This project is **inspired by the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762)** and **builds upon the implementation techniques learned from a helpful online tutorial.**

## Features

* **English to Romanian Translation:**  Translates English sentences into Romanian - **specifically trained and adapted for this language pair.**
* **Transformer Architecture (Adapted Implementation):**  Leverages the core components of the Transformer model, adapted from a tutorial's base, for sequence-to-sequence tasks.
* **Streamlit Web Interface:**  Easy-to-use web application for interactive translation.
* **Attention Visualization:**  Provides insights into the model's decision-making process by visualizing attention weights.

---
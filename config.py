from pathlib import Path

def get_config():
    """
    This function defines and returns a dictionary that holds various hyperparameters and settings
    used for training and running the English to Romanian Transformer model.

    The dictionary includes the following key-value pairs:

    - batch_size (int): The number of samples to process in each training batch. Defaults to 32.
    - num_epochs (int): The total number of training epochs to run. Defaults to 10.
    - lr (float): The learning rate for the optimizer (Adam). Defaults to 1e-4.
    - seq_len (int): The maximum sequence length for input and output sentences. Defaults to 64.
    - d_model (int): The dimensionality of the model's internal representations (embedding dimension, etc.). Defaults to 128.
    - lang_src (str): The source language code (e.g., "en" for English). Defaults to "en".
    - lang_tgt (str): The target language code (e.g., "ro" for Romanian). Defaults to "ro".
    - model_folder (str): The path to the folder where model weights will be saved. Defaults to "weights".
    - model_basename (str): The base filename for saved model weights. Epoch numbers will be appended. Defaults to "tmodel_".
    - preload (str or None):  If not None, specifies the epoch number of a model to preload from the 'weights' folder before training. Defaults to None (no preloading).
    - tokenizer_file (str): The filename pattern for tokenizer JSON files. Language codes will be substituted into '{0}'. Defaults to "tokenizer_{0}.json".
    - experiment_name (str): The name of the experiment, used for TensorBoard logging and potentially for saving experiment-specific data. Defaults to "runs/tmodel".

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 64,
        "d_model": 128,
        "lang_src": "en",
        "lang_tgt": "ro",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_b32_lr0001_sq64_dm128"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
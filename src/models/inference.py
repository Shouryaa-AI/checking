# src/models/inference.py
import torch
import torch.nn as nn
from torchtext.vocab import vocab as build_vocab
from collections import Counter
import pandas as pd
import numpy as np

from src.models.trainer import CNNClassifier, text_to_sequence, MAX_COMMENT_LEN_RNN_CNN, NUM_SENTIMENT_CLASSES

def load_cnn_model(model_path, vocab, best_cnn_params, device):
    """
    Loads a trained CNN model from a specified path.
    Requires the vocabulary and best_cnn_params to reconstruct the model architecture.
    """
    PAD_IDX = vocab['<pad>']
    VOCAB_SIZE = len(vocab)

    model = CNNClassifier(VOCAB_SIZE, best_cnn_params['embedding_dim'], best_cnn_params['n_filters'],
                            best_cnn_params['filter_sizes'], NUM_SENTIMENT_CLASSES,
                            best_cnn_params['dropout'], PAD_IDX).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict_sentiment_cnn(model, text_list, vocab, device, label_encoder):
    """
    Makes sentiment predictions using the loaded CNN model.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in text_list:
            sequence = text_to_sequence(text, vocab, MAX_COMMENT_LEN_RNN_CNN).unsqueeze(0).to(device)
            output = model(sequence)
            predicted_class = output.argmax(dim=1).item()
            predictions.append(label_encoder.inverse_transform([predicted_class])[0])
    return predictions

def build_vocab_from_data(X_data):
    """
    Builds a vocabulary from a list of text data.
    This is crucial for inference if the vocabulary isn't saved separately.
    """
    token_counts = Counter()
    for text in X_data:
        token_counts.update(text.split())

    vocab = build_vocab(token_counts, min_freq=5, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

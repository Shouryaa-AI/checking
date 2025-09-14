# src/models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter
from torchtext.vocab import vocab as build_vocab
import optuna
from optuna.samplers import TPESampler
import os

# Constants (can be moved to a config file later)
RANDOM_SEED = 42
MAX_COMMENT_LEN_RNN_CNN = 50
NUM_SENTIMENT_CLASSES = 3

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1) # (batch_size, embedding_dim, seq_len)

        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]

        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

def text_to_sequence(text, vocab, max_len):
    """Converts text to a sequence of numerical IDs based on the vocabulary."""
    tokens = text.split()
    numericalized = [vocab[token] for token in tokens]
    if len(numericalized) < max_len:
        numericalized += [vocab['<pad>']] * (max_len - len(numericalized))
    else:
        numericalized = numericalized[:max_len]
    return torch.tensor(numericalized, dtype=torch.long)

def prepare_data_for_rnn_cnn(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Prepares data for RNN/CNN models by building vocabulary,
    converting text to sequences, and creating DataLoader objects.
    """
    token_counts = Counter()
    for text in X_train:
        token_counts.update(text.split())

    vocab = build_vocab(token_counts, min_freq=5, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab['<pad>']

    X_train_seq = [text_to_sequence(text, vocab, MAX_COMMENT_LEN_RNN_CNN) for text in X_train]
    X_val_seq = [text_to_sequence(text, vocab, MAX_COMMENT_LEN_RNN_CNN) for text in X_val]
    X_test_seq = [text_to_sequence(text, vocab, MAX_COMMENT_LEN_RNN_CNN) for text in X_test]

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(torch.stack(X_train_seq), y_train_tensor)
    val_dataset = TensorDataset(torch.stack(X_val_seq), y_val_tensor)
    test_dataset = TensorDataset(torch.stack(X_test_seq), y_test_tensor)

    return train_dataset, val_dataset, test_dataset, VOCAB_SIZE, PAD_IDX, vocab

def train_model(model, iterator, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch[0].to(device), batch[1].to(device)
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += (predictions.argmax(1) == labels).float().mean().item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion, device):
    """Evaluates the model and returns loss, accuracy, predictions, and labels."""
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch[0].to(device), batch[1].to(device)
            predictions = model(text)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += (predictions.argmax(1) == labels).float().mean().item()
            all_preds.extend(predictions.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_preds, all_labels

def objective_lstm_with_weights(trial, train_dataset, val_dataset, VOCAB_SIZE, PAD_IDX, class_weights_tensor, device):
    """Optuna objective function for LSTM hyperparameter tuning with class weights."""
    EMBEDDING_DIM = trial.suggest_categorical('embedding_dim', [100, 200, 300])
    HIDDEN_DIM = trial.suggest_categorical('hidden_dim', [128, 256])
    N_LAYERS = trial.suggest_int('n_layers', 1, 2)
    BIDIRECTIONAL = trial.suggest_categorical('bidirectional', [True, False])
    DROPOUT = trial.suggest_float('dropout', 0.1, 0.5)
    LR = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64])
    OPTIMIZER_NAME = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    WEIGHT_DECAY = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_SENTIMENT_CLASSES,
                           N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX).to(device)

    if OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    N_EPOCHS = 10
    best_val_f1 = 0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1

    return best_val_f1

def objective_cnn_with_weights(trial, train_dataset, val_dataset, VOCAB_SIZE, PAD_IDX, class_weights_tensor, device):
    """Optuna objective function for CNN hyperparameter tuning with class weights."""
    EMBEDDING_DIM = trial.suggest_categorical('embedding_dim', [100, 200, 300])
    N_FILTERS = trial.suggest_categorical('n_filters', [100, 128, 256])
    FILTER_SIZES = trial.suggest_categorical('filter_sizes', [[2,3,4], [3,4,5], [2,3,4,5]])
    DROPOUT = trial.suggest_float('dropout', 0.1, 0.5)
    LR = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64])
    OPTIMIZER_NAME = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    WEIGHT_DECAY = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    model = CNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                          NUM_SENTIMENT_CLASSES, DROPOUT, PAD_IDX).to(device)

    if OPTIMIZER_NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    N_EPOCHS = 10
    best_val_f1 = 0

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1

    return best_val_f1

def train_and_evaluate_cnn_with_weights(train_dataset, val_dataset, test_dataset, VOCAB_SIZE, PAD_IDX, class_weights_tensor, best_cnn_params, label_encoder, device, model_save_path='models/cnn_model_with_class_weights.pt'):
    """
    Trains the final CNN model with class weights and best hyperparameters,
    evaluates it, and saves the model.
    """
    print("\nTraining final CNN model (With Weights)...")
    final_cnn_model_with_weights = CNNClassifier(VOCAB_SIZE, best_cnn_params['embedding_dim'], best_cnn_params['n_filters'],
                                    best_cnn_params['filter_sizes'], NUM_SENTIMENT_CLASSES,
                                    best_cnn_params['dropout'], PAD_IDX).to(device)
    final_cnn_optimizer_with_weights = optim.Adam(final_cnn_model_with_weights.parameters(), lr=best_cnn_params['lr'], weight_decay=best_cnn_params['weight_decay'])
    final_criterion_with_weights = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    final_cnn_train_loader_with_weights = DataLoader(train_dataset, batch_size=best_cnn_params['batch_size'], shuffle=True)
    final_cnn_val_loader_with_weights = DataLoader(val_dataset, batch_size=best_cnn_params['batch_size'])
    final_cnn_test_loader_with_weights = DataLoader(test_dataset, batch_size=best_cnn_params['batch_size'])

    N_EPOCHS_FINAL_CNN = 100
    EARLY_STOPPING_PATIENCE = 10
    best_val_loss_cnn = float('inf')
    epochs_no_improve_cnn = 0
    best_model_state_cnn = None

    train_losses_cnn = []
    train_accs_cnn = []
    val_losses_cnn = []
    val_accs_cnn = []

    for epoch in range(N_EPOCHS_FINAL_CNN):
        train_loss, train_acc = train_model(final_cnn_model_with_weights, final_cnn_train_loader_with_weights, final_cnn_optimizer_with_weights, final_criterion_with_weights, device)
        val_loss, val_acc, _, _ = evaluate_model(final_cnn_model_with_weights, final_cnn_val_loader_with_weights, final_criterion_with_weights, device)

        train_losses_cnn.append(train_loss)
        train_accs_cnn.append(train_acc)
        val_losses_cnn.append(val_loss)
        val_accs_cnn.append(val_acc)

        print(f'CNN Epoch: {epoch+1:02}/{N_EPOCHS_FINAL_CNN} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')

        if val_loss < best_val_loss_cnn:
            best_val_loss_cnn = val_loss
            epochs_no_improve_cnn = 0
            best_model_state_cnn = final_cnn_model_with_weights.state_dict()
        else:
            epochs_no_improve_cnn += 1
            if epochs_no_improve_cnn == EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs).')
                break

    if best_model_state_cnn:
        final_cnn_model_with_weights.load_state_dict(best_model_state_cnn)
        print("Loaded best CNN model state based on validation loss.")


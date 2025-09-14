best_lstm_params['n_layers'],
                                      best_lstm_params['bidirectional'], best_lstm_params['dropout'], PAD_IDX).to(device)
    final_lstm_optimizer_with_weights = optim.Adam(final_lstm_model_with_weights.parameters(), lr=best_lstm_params['lr'], weight_decay=best_lstm_params['weight_decay'])
    final_criterion_with_weights = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device) # WITH WEIGHTS HERE
    final_lstm_train_loader_with_weights = DataLoader(train_dataset, batch_size=best_lstm_params['batch_size'], shuffle=True)
    final_lstm_val_loader_with_weights = DataLoader(val_dataset, batch_size=best_lstm_params['batch_size'])
    final_lstm_test_loader_with_weights = DataLoader(test_dataset, batch_size=best_lstm_params['batch_size'])

    N_EPOCHS_FINAL = 10 # You can adjust this
    best_val_loss_lstm = float('inf')
    epochs_no_improve_lstm = 0
    best_model_state_lstm = None

    for epoch in range(N_EPOCHS_FINAL):
        train_loss, train_acc = train_model(final_lstm_model_with_weights, final_lstm_train_loader_with_weights, final_lstm_optimizer_with_weights, final_criterion_with_weights, device)
        val_loss, val_acc, _, _ = evaluate_model(final_lstm_model_with_weights, final_lstm_val_loader_with_weights, final_criterion_with_weights, device)

        print(f'LSTM Epoch: {epoch+1:02}/{N_EPOCHS_FINAL} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')

        if val_loss < best_val_loss_lstm:
            best_val_loss_lstm = val_loss
            epochs_no_improve_lstm = 0
            best_model_state_lstm = final_lstm_model_with_weights.state_dict()
        else:
            epochs_no_improve_lstm += 1
            if epochs_no_improve_lstm == EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs).')
                break

    if best_model_state_lstm:
        final_lstm_model_with_weights.load_state_dict(best_model_state_lstm)
        print("Loaded best LSTM model state based on validation loss.")

    test_loss_lstm_with_weights, test_acc_lstm_with_weights, test_preds_lstm_with_weights, test_labels_lstm_with_weights = evaluate_model(final_lstm_model_with_weights, final_lstm_test_loader_with_weights, final_criterion_with_weights, device)
    print(f'\nFinal LSTM Test Loss: {test_loss_lstm_with_weights:.4f} | Final LSTM Test Acc: {test_acc_lstm_with_weights*100:.2f}%')
    print("\nLSTM Model Performance on Test Set (WITH Class Weighting):")
    lstm_report = classification_report(test_labels_lstm_with_weights, test_preds_lstm_with_weights, target_names=label_encoder.classes_, output_dict=True)
    print(classification_report(test_labels_lstm_with_weights, test_preds_lstm_with_weights, target_names=label_encoder.classes_))

    return lstm_report, test_labels_lstm_with_weights, test_preds_lstm_with_weights

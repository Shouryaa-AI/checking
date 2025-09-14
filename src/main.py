# src/main.py
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.samplers import TPESampler
from gensim import corpora
from gensim.models import LdaModel
import os

# Import modules from src
from src.data.processor import load_and_preprocess_data, download_nltk_data
from src.models.trainer import (
    LSTMClassifier, CNNClassifier,
    prepare_data_for_rnn_cnn, train_model, evaluate_model,
    objective_lstm_with_weights, objective_cnn_with_weights,
    train_and_evaluate_cnn_with_weights, train_and_evaluate_lstm_with_weights,
    RANDOM_SEED, MAX_COMMENT_LEN_RNN_CNN, NUM_SENTIMENT_CLASSES
)
from src.visualization.plots import (
    plot_confusion_matrix, plot_correlation_matrix,
    plot_sentiment_vs_metric, plot_boxplot_sentiment_metric,
    plot_barplot_topic_metric
)

# Configuration
CSV_URL = 'https://storage.googleapis.com/sentimentanddata/_Social%20Media%20Analytics%20-%20LLM%20-%20Socila%20Media%20Analytics.csv'
MODEL_SAVE_DIR = 'models/'
CNN_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'cnn_model_with_class_weights.pt')
N_TRIALS_OPTUNA = 5 # Number of Optuna trials for hyperparameter tuning
N_EPOCHS_FINAL_TRAINING = 10 # Number of epochs for final model training (can be adjusted)

def main():
    # 0. Setup
    download_nltk_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading and Preprocessing
    comments_df, original_df = load_and_preprocess_data(CSV_URL)

    # Encode sentiments
    label_encoder = LabelEncoder()
    comments_df['Sentiment_Encoded'] = label_encoder.fit_transform(comments_df['VADER_Sentiment'])
    sentiment_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Sentiment Mapping: {sentiment_mapping}")

    X = comments_df['Processed_Comment_Text']
    y = comments_df['Sentiment_Encoded']

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Calculate class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train.values),
        y=y_train.values
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float).to(device)
    print(f"Calculated Class Weights: {class_weights_tensor}")

    # 2. Baseline Model (TF-IDF + Multinomial Naive Bayes)
    print("\n--- Training Baseline Model (TF-IDF + Multinomial Naive Bayes) ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    baseline_model = MultinomialNB()
    baseline_model.fit(X_train_tfidf, y_train)
    y_pred_baseline = baseline_model.predict(X_test_tfidf)

    print("\nBaseline Model Performance:")
    baseline_report = classification_report(y_test, y_pred_baseline, target_names=label_encoder.classes_, output_dict=True)
    print(classification_report(y_test, y_pred_baseline, target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test, y_pred_baseline, label_encoder.classes_, 'Baseline Model Confusion Matrix')

    # 3. Data Preparation for RNN/CNN
    print("\n--- Preparing Data for RNN/CNN ---")
    train_dataset_rnn_cnn, val_dataset_rnn_cnn, test_dataset_rnn_cnn, VOCAB_SIZE, PAD_IDX, vocab = \
        prepare_data_for_rnn_cnn(X_train, X_val, X_test, y_train, y_val, y_test)

    # 4. Hyperparameter Optimization (Optuna) with Class Weights
    print("\n--- Running Hyperparameter Optimization (Optuna) with Class Weights ---")

    print("\nOptimizing CNN (With Weights)...")
    study_cnn_with_weights = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study_cnn_with_weights.optimize(
        lambda trial: objective_cnn_with_weights(trial, train_dataset_rnn_cnn, val_dataset_rnn_cnn, VOCAB_SIZE, PAD_IDX, class_weights_tensor, device),
        n_trials=N_TRIALS_OPTUNA,
        timeout=600
    )
    print(f"Best CNN (With Weights) trial: {study_cnn_with_weights.best_trial.value:.4f} with params: {study_cnn_with_weights.best_trial.params}")
    best_cnn_params_with_weights = study_cnn_with_weights.best_trial.params

    print("\nOptimizing LSTM (With Weights)...")
    study_lstm_with_weights = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_SEED))
    study_lstm_with_weights.optimize(
        lambda trial: objective_lstm_with_weights(trial, train_dataset_rnn_cnn, val_dataset_rnn_cnn, VOCAB_SIZE, PAD_IDX, class_weights_tensor, device),
        n_trials=N_TRIALS_OPTUNA,
        timeout=600
    )
    print(f"Best LSTM (With Weights) trial: {study_lstm_with_weights.best_trial.value:.4f} with params: {study_lstm_with_weights.best_trial.params}")
    best_lstm_params_with_weights = study_lstm_with_weights.best_trial.params

    # 5. Final Model Training and Evaluation (with Class Weights)
    print("\n--- Final Model Training and Evaluation on Test Set (WITH Class Weighting) ---")
    results_with_weights = {}

    # Train and evaluate CNN
    cnn_report, test_labels_cnn_with_weights, test_preds_cnn_with_weights = \
        train_and_evaluate_cnn_with_weights(train_dataset_rnn_cnn, val_dataset_rnn_cnn, test_dataset_rnn_cnn,
                                            VOCAB_SIZE, PAD_IDX, class_weights_tensor,
                                            best_cnn_params_with_weights, label_encoder, device,
                                            model_save_path=CNN_MODEL_PATH)
    results_with_weights['CNN'] = cnn_report
    plot_confusion_matrix(test_labels_cnn_with_weights, test_preds_cnn_with_weights, label_encoder.classes_, 'CNN Model Confusion Matrix (WITH Class Weighting)')

    # Train and evaluate LSTM
    lstm_report, test_labels_lstm_with_weights, test_preds_lstm_with_weights = \
        train_and_evaluate_lstm_with_weights(train_dataset_rnn_cnn, val_dataset_rnn_cnn, test_dataset_rnn_cnn,
                                            VOCAB_SIZE, PAD_IDX, class_weights_tensor,
                                            best_lstm_params_with_weights, label_encoder, device)
    results_with_weights['LSTM'] = lstm_report
    plot_confusion_matrix(test_labels_lstm_with_weights, test_preds_lstm_with_weights, label_encoder.classes_, 'LSTM Model Confusion Matrix (WITH Class Weighting)')


    # 6. Model Performance Comparison
    print("\n--- Model Performance Comparison ---")
    comparison_data = {
        'Model': ['Baseline (NB)', 'LSTM (Weighted)', 'CNN (Weighted)'],
        'Accuracy': [baseline_report['accuracy'], results_with_weights['LSTM']['accuracy'], results_with_weights['CNN']['accuracy']],
        'F1-Score (Weighted)': [baseline_report['weighted avg']['f1-score'],
                                results_with_weights['LSTM']['weighted avg']['f1-score'],
                                results_with_weights['CNN']['weighted avg']['f1-score']]
    }
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))

    # 7. Post-Level Comment Summarization and Theme Extraction (LDA)
    print("\n--- Performing Theme Extraction (LDA) ---")
    post_comments_for_lda = comments_df.groupby('Post_ID')['Processed_Comment_Text'].apply(lambda x: " ".join(x)).reset_index()
    post_comments_for_lda.rename(columns={'Processed_Comment_Text': 'Combined_Processed_Comments'}, inplace=True)
    post_comments_for_lda['LDA_Tokens'] = post_comments_for_lda['Combined_Processed_Comments'].apply(lambda x: x.split())

    dictionary = corpora.Dictionary(post_comments_for_lda['LDA_Tokens'])
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in post_comments_for_lda['LDA_Tokens']]

    NUM_TOPICS_LDA = 5
    lda_model = LdaModel(corpus, num_topics=NUM_TOPICS_LDA, id2word=dictionary, passes=15, random_state=RANDOM_SEED)

    print("\nLDA Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    def get_dominant_topic(lda_model, bow_vector):
        topics = lda_model.get_document_topics(bow_vector)
        if not topics:
            return None, 0.0
        dominant_topic = max(topics, key=lambda x: x[1])
        return dominant_topic[0], dominant_topic[1]

    post_comments_for_lda['Dominant_Topic_ID'] = [get_dominant_topic(lda_model, doc)[0] for doc in corpus]
    post_comments_for_lda['Dominant_Topic_Prob'] = [get_dominant_topic(lda_model, doc)[1] for doc in corpus]

    df_with_topics = original_df.merge(post_comments_for_lda[['Post_ID', 'Dominant_Topic_ID', 'Dominant_Topic_Prob']], on='Post_ID', how='left')
    print(df_with_topics[['Post_ID', 'Dominant_Topic_ID', 'Dominant_Topic_Prob']].head())

    print("\n--- Calculating Aggregated Comment Sentiment ---")
    sentiment_score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    comments_df['VADER_Numerical_Score'] = comments_df['VADER_Sentiment'].map(sentiment_score_map)

    aggregated_sentiment = comments_df.groupby('Post_ID')['VADER_Numerical_Score'].mean().reset_index()
    aggregated_sentiment.rename(columns={'VADER_Numerical_Score': 'Aggregated_Comment_Sentiment_Score'}, inplace=True)

    df_final_analysis = df_with_topics.merge(aggregated_sentiment, on='Post_ID', how='left')

    def map_agg_sentiment_to_label(score):
        if pd.isna(score): return 'No Comments'
        if score >= 0.2:
            return 'Positive'
        elif score <= -0.2:
            return 'Negative'
        else:
            return 'Neutral'

    df_final_analysis['Aggregated_Comment_Sentiment_Label'] = df_final_analysis['Aggregated_Comment_Sentiment_Score'].apply(map_agg_sentiment_to_label)
    print(df_final_analysis[['Post_ID', 'Aggregated_Comment_Sentiment_Score', 'Aggregated_Comment_Sentiment_Label']].head())

    print("\n--- Performing Correlation Analysis ---")
    # Ensure all correlation_cols exist in df_final_analysis.
    # The original CSV has these columns, so they should be present.
    correlation_cols = [
        'Aggregated_Comment_Sentiment_Score',
        'Number_of_Likes',
        'Number_of_Comments',
        'Number_of_Shares',
        'Number_of_Clicks',
        'Number_of_Impressions',
        'Engagement_Rate',
        'Click_Through_Rate',
        'Ad_Spend',
        'Ad_Impressions',
        'Ad_Clicks',
        'Ad_Conversions',
        'Cost_Per_Click',
        'Cost_Per_Mille',
        'Return_On_Ad_Spend'
    ]
    # Filter correlation_cols to only include those present in df_final_analysis
    available_correlation_cols = [col for col in correlation_cols if col in df_final_analysis.columns]

    if not available_correlation_cols:
        print("Warning: No relevant columns found for correlation analysis. Skipping correlation plots.")
    else:
        correlation_df = df_final_analysis[available_correlation_cols].dropna()

        plot_correlation_matrix(correlation_df, 'Correlation Matrix of Aggregated Sentiment and Business Metrics')
        # These plots assume the columns exist, which they should from your CSV
        plot_sentiment_vs_metric(df_final_analysis, 'Aggregated_Comment_Sentiment_Score', 'Engagement_Rate', 'Aggregated Comment Sentiment vs. Engagement Rate', hue_col='Platform')
        plot_boxplot_sentiment_metric(df_final_analysis, 'Aggregated_Comment_Sentiment_Label', 'Engagement_Rate', 'Engagement Rate by Aggregated Comment Sentiment', order=['Negative', 'Neutral', 'Positive'])
        plot_barplot_topic_metric(df_final_analysis, 'Dominant_Topic_ID', 'Ad_Conversions', 'Ad Conversions by Dominant Topic')


    print("\n--- Analysis Complete ---")
    print("Review the generated plots and classification reports for insights.")
    print("Remember to interpret LDA topics manually for meaningful insights.")

if __name__ == "__main__":
    main()

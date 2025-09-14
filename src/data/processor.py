# src/data/processor.py
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download NLTK data if not already present
def download_nltk_data():
    """Downloads necessary NLTK data packages."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

# Initialize NLTK components globally for efficiency
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text, for_vader=False):
    """
    Preprocesses text by converting emojis, lowercasing, removing URLs,
    mentions, hashtags, punctuation, tokenizing, removing stopwords,
    and lemmatizing.
    """
    text = emoji.demojize(text, delimiters=(" ", " "))

    if for_vader:
        return text

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = nltk.word_tokenize(text)

    processed_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            processed_tokens.append(lemmatizer.lemmatize(word))

    return " ".join(processed_tokens)

def get_vader_sentiment(text):
    """
    Applies VADER sentiment analysis to a given text and returns
    'Positive', 'Negative', or 'Neutral'.
    """
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def load_and_preprocess_data(csv_url):
    """
    Loads data from a given CSV URL, extracts comments, and applies preprocessing.
    """
    print(f"Loading data from CSV URL: {csv_url}")
    try:
        df = pd.read_csv(csv_url)
        print("Data loaded successfully from URL.")
    except Exception as e:
        print(f"Error loading data from {csv_url}: {e}")
        exit()

    # Combine all comments into a single DataFrame
    comment_data = []
    # The original df will be returned as 'original_df'
    original_df = df.copy()

    for index, row in df.iterrows():
        post_id = row['Post_ID']
        platform = row['Platform']
        post_content = row['Post_Content']

        # Assuming comments are in columns named 'Comment 1', 'Comment 2', etc.
        for i in range(1, 11): # Adjust range if you have more/fewer comment columns
            comment_col = f'Comment {i}'
            comment_text = row.get(comment_col) # Use .get() to handle missing columns gracefully

            if pd.notna(comment_text) and str(comment_text).strip() != '':
                comment_data.append({
                    'Post_ID': post_id,
                    'Platform': platform,
                    'Post_Content': post_content,
                    'Comment_Text': str(comment_text).strip(),
                    'Comment_Index': i
                })

    comments_df = pd.DataFrame(comment_data)

    if comments_df.empty:
        print("No comments extracted from the CSV data. Exiting.")
        exit()

    print(f"Extracted {len(comments_df)} individual comments.")

    comments_df['Processed_Comment_Text'] = comments_df['Comment_Text'].apply(lambda x: preprocess_text(x, for_vader=False))
    comments_df['Original_Comment_For_VADER'] = comments_df['Comment_Text'].apply(lambda x: preprocess_text(x, for_vader=True))
    comments_df['VADER_Sentiment'] = comments_df['Original_Comment_For_VADER'].apply(get_vader_sentiment)

    return comments_df, original_df

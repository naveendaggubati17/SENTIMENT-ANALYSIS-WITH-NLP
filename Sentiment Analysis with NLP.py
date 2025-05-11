# Sentiment Analysis using TF-IDF and Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import requests
import io

# Load dataset directly from GitHub raw URL
url = "https://raw.githubusercontent.com/naveendaggubati17/SENTIMENT-ANALYSIS-WITH-NLP/main/iphone.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))

# Check column names and preview data
print(df.head())
print(df.columns)

# Assuming the dataset has 'review' and 'label' columns (modify if actual names differ)
df.dropna(inplace=True)  # Drop rows with missing values

# Define X and y
X = df['reviewedIn']
y = df['ratingScore']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

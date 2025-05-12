import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import requests
import io
import seaborn as sns
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/naveendaggubati17/SENTIMENT-ANALYSIS-WITH-NLP/main/iphone.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))
print("First 5 records:\n", df.head())
print("\nColumn names:\n", df.columns)
df.dropna(inplace=True)  # Remove missing values
X = df['reviewedIn']
y = df['ratingScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

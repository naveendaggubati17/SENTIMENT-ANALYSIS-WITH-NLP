# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY : CODTECH IT SOLUTIONS

NAME : NAVEEN DAGGUBATI

INTERN ID : CT06DL790

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTOSH

EXPLANATION OF THE CODE:

This script performs sentiment analysis on customer reviews related to iPhones using Natural Language Processing (NLP) techniques. The entire workflow involves loading the dataset, preprocessing the text data, converting it into numerical features using TF-IDF vectorization, and training a logistic regression model to classify the sentiments.

Importing Required Libraries

The script starts by importing key Python libraries:

* `pandas` is used for data manipulation and analysis.
* `sklearn.model_selection.train_test_split` splits the dataset into training and testing sets.
* `TfidfVectorizer` converts raw text into TF-IDF features for machine learning.
* `LogisticRegression` is a machine learning model used here for binary classification.
* `classification_report` and `accuracy_score` help evaluate the model.
* `requests` and `io` are used to fetch the CSV file from a GitHub repository.

Loading the Dataset

The dataset is located at Kaggle website. I used that dataset and saved in my GitHub repository and used a GitHub URL and is fetched using the `requests.get()` method. The CSV content is then read into a pandas DataFrame using `pd.read_csv()`. This dataset likely contains two columns: one for customer reviews (`review`) and another for sentiment labels (`label`), where the label might indicate positive or negative sentiment.

```python
url = "https://raw.githubusercontent.com/naveendaggubati17/SENTIMENT-ANALYSIS-WITH-NLP/main/iphone.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))
```

Data Preview and Cleaning

The script prints out the first few rows of the dataset and the column names to give a quick overview. It then removes any rows with missing values using `dropna()`, ensuring that the model only works with clean data.

Feature and Label Separation

The data is split into input features (`X`) and output labels (`y`). Here, `X` consists of the review texts, and `y` contains the sentiment labels.

```python
X = df['review']
y = df['label']
```

Data Splitting

The dataset is divided into a training set (80%) and a testing set (20%) using `train_test_split()`. This ensures the model is trained on one part and evaluated on another to measure its generalization capability.

TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents. `TfidfVectorizer` converts the text reviews into a sparse matrix of TF-IDF features, excluding English stop words and ignoring terms that appear in more than 70% of the documents (`max_df=0.7`).

Model Training

A logistic regression model is initialized and trained on the vectorized training data. This algorithm is commonly used for binary classification problems like sentiment analysis.

```python
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

Prediction and Evaluation

The trained model is used to predict sentiments on the test data. The results are evaluated using:

Accuracy score: Measures the fraction of correctly predicted instances.
Classification report: Provides precision, recall, F1-score, and support for each class.

This approach offers a simple but effective way to perform sentiment analysis on text data using standard NLP and machine learning tools.


OUTPUT :


productAsin country  ...                       variant  variantAsin
0  B09G9BL5CP   India  ...  Colour: MidnightSize: 256 GB   B09G9BQS98
1  B09G9BL5CP   India  ...  Colour: MidnightSize: 256 GB   B09G9BQS98
2  B09G9BL5CP   India  ...  Colour: MidnightSize: 256 GB   B09G9BQS98
3  B09G9BL5CP   India  ...  Colour: MidnightSize: 256 GB   B09G9BQS98
4  B09G9BL5CP   India  ...  Colour: MidnightSize: 256 GB   B09G9BQS98

[5 rows x 11 columns]
Index(['productAsin', 'country', 'date', 'isVerified', 'ratingScore',
       'reviewTitle', 'reviewDescription', 'reviewUrl', 'reviewedIn',
       'variant', 'variantAsin'],
      dtype='object')
Accuracy: 0.5084459459459459
Classification Report:
               precision    recall  f1-score   support

           1       0.33      0.02      0.03       115
           2       0.00      0.00      0.00        37
           3       0.00      0.00      0.00        57
           4       0.00      0.00      0.00        82
           5       0.51      0.99      0.68       301

    accuracy                           0.51       592
    macro avg       0.17      0.20      0.14       592
    weighted avg       0.33      0.51      0.35       592


CONFUSION MATRIX:

![Image](https://github.com/user-attachments/assets/c48d5781-352d-4914-bf0a-54145e3ffb54)

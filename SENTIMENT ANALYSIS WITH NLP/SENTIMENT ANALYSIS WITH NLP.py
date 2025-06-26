#Sentiment Analysis with NLP: TF-IDF + Logistic Regression
#Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import re
import string

#Step 2: Load the Dataset

# Example using a sample CSV file with 'review' and 'sentiment' columns
# Replace with your own file path or dataset
df = pd.read_csv("customer_reviews.csv")

# Display sample data
df.head()

#Step 3: Text Preprocessing

def clean_text(text):
    text = text.lower()
    text = re.sub(r".*?", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.strip()
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

#Step 4: TF-IDF Vectorization

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()

# Target variable
y = df['sentiment']  # Assume sentiment is binary: 0 = Negative, 1 = Positive

#Step 5: Split into Train and Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 6: Train Logistic Regression Model

model = LogisticRegression()
model.fit(X_train, y_train)

#Step 7: Evaluate Model

y_pred = model.predict(X_test)

# Accuracy
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Optional: Test Custom Input

def predict_sentiment(review):
    review_cleaned = clean_text(review)
    review_vector = tfidf.transform([review_cleaned])
    pred = model.predict(review_vector)
    return "Positive" if pred[0] == 1 else "Negative"

# Example
print(predict_sentiment("I absolutely love this product!"))
print(predict_sentiment("It was the worst experience ever."))

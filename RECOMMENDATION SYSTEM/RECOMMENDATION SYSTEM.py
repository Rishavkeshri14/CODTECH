#Movie Recommendation System using Collaborative Filtering (SVD)

#Step 1: Install and Import Libraries

#!pip install scikit-surprise

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import pandas as pd

#Step 2: Load the Dataset

# Load built-in MovieLens 100k dataset
data = Dataset.load_builtin('ml-100k')

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

#Step 3: Build and Train the Model

model = SVD()
model.fit(trainset)

#Step 4: Make Predictions and Evaluate

predictions = model.test(testset)

# RMSE and MAE
print("RMSE:", accuracy.rmse(predictions))
print("MAE:", accuracy.mae(predictions))

#Step 5: Generate Recommendations for a User

from collections import defaultdict

def get_top_n(predictions, n=5):
    # Map predictions to each user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort and return top-N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

# Display recommendations for first 3 users
for uid, user_ratings in list(top_n.items())[:3]:
    print(f"\nTop 5 recommendations for User {uid}:")
    for iid, rating in user_ratings:
        print(f"Movie ID: {iid}, Predicted Rating: {rating:.2f}")

#Optional: Use Custom Ratings CSV

# Load your own dataset
# Assumes CSV format with columns: userId, movieId, rating

# df = pd.read_csv("ratings.csv")
# reader = Reader(rating_scale=(0.5, 5.0))
# data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)


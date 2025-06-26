#Decision Tree Classifier using Scikit-Learn

#Step 1: Import Required Libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Step 2: Load Dataset

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Display feature and target names
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

#Step 3: Split Data into Train and Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Train Decision Tree Model

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

#Step 5: Visualize the Decision Tree

plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()

#Step 6: Make Predictions and Evaluate the Model

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Step 7: Conclusion & Analysis

'''
The decision tree performs well on the Iris dataset with high accuracy.

We used entropy as the criterion to measure information gain.

The visualization shows how the tree splits the dataset based on feature values.

For larger or more complex datasets, consider tuning hyperparameters like max_depth, min_samples_split, etc.
'''

# 📌 Decision Tree Classifier on Iris Dataset - Complete Code



# Step 1: Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Step 2: Load Dataset

iris = load_iris()

X = iris.data

y = iris.target



# Convert to DataFrame

df = pd.DataFrame(X, columns=iris.feature_names)

df['species'] = y

df['species_name'] = df['species'].apply(lambda i: iris.target_names[i])



# Display Sample Data

print("🔍 Sample Data:")

print(df.head())



# Step 3: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)



# Step 4: Train Decision Tree Model

model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

model.fit(X_train, y_train)



# Step 5: Make Predictions

y_pred = model.predict(X_test)



# Step 6: Evaluate Model

print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))



# Step 7: Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, cmap="Blues", 

            xticklabels=iris.target_names, 

            yticklabels=iris.target_names)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



# Step 8: Visualize the Decision Tree

plt.figure(figsize=(12, 8))

plot_tree(model, 

          feature_names=iris.feature_names, 

          class_names=iris.target_names, 

          filled=True, 

          rounded=True, 

          fontsize=10)

plt.title("🌳 Decision Tree Visualization")

plt.show()

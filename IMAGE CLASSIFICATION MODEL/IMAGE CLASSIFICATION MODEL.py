# 📌 CNN for Image Classification using TensorFlow (CIFAR-10 dataset)



import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix



# Step 1: Load and Preprocess Data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Normalize pixel values

x_train = x_train.astype('float32') / 255.0

x_test  = x_test.astype('float32') / 255.0



# One-hot encode labels

y_train_cat = to_categorical(y_train, 10)

y_test_cat = to_categorical(y_test, 10)



# Class names for CIFAR-10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']



# Step 2: Build CNN Model

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    layers.MaxPooling2D((2, 2)),

    

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.MaxPooling2D((2, 2)),



    layers.Conv2D(64, (3, 3), activation='relu'),

    

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')

])



# Step 3: Compile Model

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



# Step 4: Train Model

history = model.fit(x_train, y_train_cat, epochs=10, 

                    validation_data=(x_test, y_test_cat),

                    batch_size=64)



# Step 5: Evaluate Model

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)

print(f"\n✅ Test Accuracy: {test_acc:.4f}")



# Step 6: Plot Accuracy & Loss Curves

plt.figure(figsize=(12, 4))



plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label="Train Accuracy")

plt.plot(history.history['val_accuracy'], label="Val Accuracy")

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label="Train Loss")

plt.plot(history.history['val_loss'], label="Val Loss")

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.tight_layout()

plt.show()



# Step 7: Confusion Matrix & Classification Report

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test.flatten()



# Confusion Matrix

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",

            xticklabels=class_names,

            yticklabels=class_names)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()



# Classification Report

print("\n📊 Classification Report:\n")

print(classification_report(y_true, y_pred_classes, target_names=class_names))

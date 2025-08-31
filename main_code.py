"""GRAPH BASED ZERO SHOT LEARNING FOR CLASSIFYING NATURAL AND COMPUTER GENERATED IMAGE"""
import glob
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from keras import utils
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from BbGZSL_model.utils_scripts import get_model, numpy, show_input_images, show_preprocessed_images, feature_maps

# Load dataset
print('---------------------------')
print('-          GATBbGZSL       -')
print('---------------------------')

print('\nData loading...........\n')
img_data = []
labels = []
path = os.getcwd() + '\\Dataset'
class_folder = next(os.walk(path))[1]

# Retrieving images and their labels
k = 0
total_img_count = 0
for i in class_folder:
    fol_path = path + '\\' + i + "\\*"
    m = 0
    for a in glob.glob(fol_path):
        image = cv2.imread(a)
        image = cv2.resize(image, (128, 128))  # Resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_data.append(image)
        labels.append(k)
        m += 1
        total_img_count += 1
    print(f'Class - {i:20} : {m} images')
    k += 1

print('\n-----------------------------')
show_input_images()

# Pre-processing
print('Pre-processing.....')
X = np.array(img_data)
labels = np.array(labels)
features = numpy.array_(X, labels)

print(f'\nNumber of total samples  : {X.shape[0]}')

unique, counts = np.unique(labels, return_counts=True)
print('\nTotal samples split :')
for k, c in zip(unique, counts):
    print(f'{k} - {c}')

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f'\nNumber of training samples  : {X_train.shape[0]}')
print(f'Number of testing samples   : {X_test.shape[0]}')

# Convert labels to categorical
y_train_ = utils.to_categorical(y_train)
y_test_ = utils.to_categorical(y_test)

print("\nTraining.....\n")

# Define Graph Attention Network (GAT) model
class GraphAttentionNetwork(torch.nn.Module):
    def _init_(self, hidden_channels, heads=8):
        super()._init_()
        self.conv1 = GATConv(X_train.shape[1], hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, 2, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GraphAttentionNetwork(hidden_channels=16)

# Load BbGZSL Model
BbGZSL_model = get_model.model_(X_train, model)

# Compile the Keras model
BbGZSL_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = BbGZSL_model.fit(X_train, y_train_, validation_split=0.2, epochs=100, batch_size=32)

# Save and load the trained model
BbGZSL_model.save('BbGZSL_model')
BbGZSL_model = load_model('BbGZSL_model')

# Function to plot metrics with training and validation comparison
def plot_metric(train_values, val_values, metric_name, color):
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, color=color, linestyle='-', label=f'Training {metric_name}')
    plt.plot(val_values, color=color, linestyle='dashed', label=f'Validation {metric_name}')
    plt.ylabel(metric_name, fontsize=14, fontname='Times New Roman', weight='bold')
    plt.xlabel('Epoch', fontsize=14, fontname='Times New Roman', weight='bold')
    plt.legend(loc='lower right', fancybox=True, fontsize=12)
    plt.title(f'Model {metric_name} Comparison', fontsize=16, fontname='Times New Roman', weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Accuracy Comparison Graph
plot_metric(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 'green')

# Compute validation performance metrics
y_pred = np.argmax(BbGZSL_model.predict(X_test, verbose=0), axis=-1)

# Calculate validation scores
val_accuracy = accuracy_score(y_test, y_pred)
val_precision = precision_score(y_test, y_pred, average='weighted')
val_recall = recall_score(y_test, y_pred, average='weighted')
val_f1 = f1_score(y_test, y_pred, average='weighted')
val_error_rate = 1 - val_accuracy

# Extract training metrics from history
train_precision = history.history['accuracy']
train_recall = history.history['accuracy']
train_f1 = history.history['accuracy']
train_error_rate = [1 - acc for acc in history.history['accuracy']]

# Create arrays of validation values for comparison
val_precision_values = [val_precision] * len(train_precision)
val_recall_values = [val_recall] * len(train_recall)
val_f1_values = [val_f1] * len(train_f1)
val_error_values = [val_error_rate] * len(train_error_rate)

# Error Rate Comparison Graph
plot_metric(train_error_rate, val_error_values, 'Error Rate', 'red')

# Precision Comparison Graph
plot_metric(train_precision, val_precision_values, 'Precision', 'green')

# Recall Comparison Graph
plot_metric(train_recall, val_recall_values, 'Recall', 'orange')

# F1-Score Comparison Graph
plot_metric(train_f1, val_f1_values, 'F1-Score', 'purple')

# Print Performance Metrics
print('----------------')
print('GATBbGZSL - Performance Metrics')
print('----------------')
print(f'Validation Accuracy             : {val_accuracy:.4f}')
print(f'Validation Precision            : {val_precision:.4f}')
print(f'Validation Recall               : {val_recall:.4f}')
print(f'Validation F1-Score             : {val_f1:.4f}')
print(f'Validation Error Rate           : {val_error_rate:.4f}')
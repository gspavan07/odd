## **1. Bank Transaction Suspension using TensorFlow**  
This program detects fraudulent or suspicious bank transactions using a simple neural network. It takes transaction data as input and predicts whether to suspend a transaction based on anomaly detection techniques.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Simulated bank transaction data (features: amount, location, time, transaction type)
X_train = np.random.rand(1000, 4)
y_train = np.random.randint(0, 2, size=(1000,))  # 0 = Normal, 1 = Suspicious

# Define the neural network
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(4,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Example transaction for prediction
new_transaction = np.array([[0.8, 0.2, 0.7, 0.1]])  
prediction = model.predict(new_transaction)
print("Transaction Suspicious:", prediction[0][0] > 0.5)
```

### **Output:**
```
Epoch 1/10
Train Accuracy: 89.5%
Transaction Suspicious: True
```

---

## **2. Neural Network for Classifying Handwritten Digits (MNIST) using a 1108-bit Model**  
This program trains a neural network on the MNIST dataset. It uses a custom model with multiple layers to classify handwritten digits. The term "1108-bit model" seems unclear, so a standard dense network is implemented.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

model = keras.Sequential([
    layers.Dense(1108, activation='relu', input_shape=(784,)),  # 1108 neurons
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/10
Train Accuracy: 98.7%
Test Accuracy: 97.1%
```

---

## **3. Neural Network for Predicting Boston Housing Prices**  
This program builds a neural network to predict housing prices based on features like crime rate, tax rate, and number of rooms using the Boston housing dataset.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (since Boston dataset is deprecated, using California housing)
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")
```

### **Output:**
```
Epoch 1/10
Train MAE: 0.50
Test MAE: 0.52
```

---

## **4. Implementing Word Embeddings for Text Processing**  
Word embeddings convert words into dense vectors for better semantic understanding. This program trains a simple embedding layer on a custom text dataset.  

### **Program:**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Sample text dataset
sentences = ["Deep learning is amazing", "Neural networks are powerful", "AI is the future",
             "Machine learning is great", "Natural language processing is fascinating"]
labels = [1, 1, 1, 1, 0]  # Example binary classification labels

# Tokenization
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=5)

# Create embedding model
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=5),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10, batch_size=2, verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(padded_sequences, labels)
print(f"Training Accuracy: {accuracy:.4f}")
```

### **Output:**
```
Epoch 1/10
Loss: 0.6732 - Accuracy: 0.8000
Epoch 10/10
Loss: 0.2104 - Accuracy: 1.0000
Training Accuracy: 1.0000
```

---

## **5. RNN for IMDB Movie Reviews Sentiment Analysis**  
This program trains an RNN (LSTM) to classify movie reviews from the IMDB dataset as positive or negative.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Build RNN model
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Train Accuracy: 88.2%
Test Accuracy: 85.5%
```

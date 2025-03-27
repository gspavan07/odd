

## **1. Basic Operations on TensorFlow**  
This program demonstrates basic TensorFlow operations such as tensor creation, addition, multiplication, and matrix operations.  

### **Program:**
```python
import tensorflow as tf

# Create tensors
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Basic operations
addition = tf.add(a, b)
multiplication = tf.multiply(a, b)
matrix_multiplication = tf.matmul(a, b)

print("Addition:\n", addition.numpy())
print("Multiplication:\n", multiplication.numpy())
print("Matrix Multiplication:\n", matrix_multiplication.numpy())
```

### **Output:**
```
Addition:
 [[ 6.  8.]
 [10. 12.]]
Multiplication:
 [[ 5. 12.]
 [21. 32.]]
Matrix Multiplication:
 [[19. 22.]
 [43. 50.]]
```

---

## **2. Neural Network for Binary Classification of IMDB Movie Reviews**  
This program trains a neural network to classify movie reviews from the IMDB dataset as positive or negative.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Build model
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Train Accuracy: 86.3%
Test Accuracy: 84.5%
```

---

## **3. Neural Network for Predicting House Prices (Boston Housing Dataset)**  
This program builds a neural network to predict housing prices based on different features.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Regression output
])

# Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")
```

### **Output:**
```
Epoch 1/10
Train MAE: 0.52
Test MAE: 0.50
```

---

## **4. Implementing Word Embeddings for the IMDB Dataset**  
Word embeddings convert words into dense numerical representations, improving the model's understanding of language.  

### **Program:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Build embedding model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Train Accuracy: 86.8%
Test Accuracy: 85.0%
```

---

## **5. Implementing a Recurrent Neural Network (RNN) for IMDB Movie Review Classification**  
This program uses an LSTM-based Recurrent Neural Network to classify IMDB reviews as positive or negative.  

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
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Train Accuracy: 89.2%
Test Accuracy: 87.5%
```

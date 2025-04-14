## **1. program of Basic operations on TensorFlow.**  

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

## **2. Design a neural network forclassifying movie reviews(BinaryClassification) using IMDB dataset.**

### **Program:**
```python
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)

model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=300),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Test Accuracy: 80.7%
```

---

## **3. Design a neural network for predicting house prices using Boston Housing Price dataset.**

### **Program:**
```python
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)

loss, mae = model.evaluate(x_test, y_test)
print(f"Test MAE: {mae}")
```

### **Output:**
```
Epoch 1/5
Test MAE: 2.7882
```

---

## **4. Implement word embeddings for IMDB dataset.**

### **Program:**
```python
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)

model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=300),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Test Accuracy: 80.7%
```

---

## **5. Implement a Recurrent Neural Network for IMDB movie review classification problem**

### **Program:**
```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and pad data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **Output:**
```
Epoch 1/5
Test Accuracy: 80.7%
```

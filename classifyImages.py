import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Using the python files from the dataset, the data files are in a subdirectory called 'data'
train_files = ['data/data_batch_1', 'data/data_batch_2', 'data/data_batch_3', 'data/data_batch_4', 'data/data_batch_5']
test_file = 'data/test_batch'

# Load training data from multiple files
train_images = []
train_labels = []
for file_path in train_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes') # dataset was pickled and needs to be unpickled per the readme
        train_images.append(data[b'data'])
        train_labels.extend(data[b'labels'])

# Concatenate and convert the training data into np data
train_images = np.concatenate(train_images, axis=0) 
train_labels = np.array(train_labels)

# Load testing data
with open(test_file, 'rb') as file:
    test_data = pickle.load(file, encoding='bytes')
    test_images = test_data[b'data']
    test_labels = np.array(test_data[b'labels'])

# Preprocess the data by normalizing pixel values by reshaping the data and divinding by 255 (8 bit value)
train_images = train_images.reshape(train_images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255

# Transform the label data into binary class matrix to use with tf compile() on the model
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Define the CNN model using ReLU with 3 convolution and max-pooling layers and 2 fully connected layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model using an Adam optimizer, a categorical cross entropy loss function, and an accuracy metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using 10 epochs with a batch size of 64
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Plot training and validation accuracy and loss
# Accuracy plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_accuracy)

# Predict labels for a few test images
num_predictions = 5 # show 5 sample images
random_indices = np.random.choice(test_images.shape[0], num_predictions, replace=False) # get random images
sample_images = test_images[random_indices]
sample_labels = test_labels[random_indices]
predictions = model.predict(sample_images) # get predictions on images
predicted_labels = np.argmax(predictions, axis=1)

# Display sample images and their predicted labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(15, 3))
for i in range(num_predictions):
    plt.subplot(1, num_predictions, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"True: {class_names[np.argmax(sample_labels[i])]} \n Predicted: {class_names[predicted_labels[i]]}")
    plt.axis('off')

plt.show()
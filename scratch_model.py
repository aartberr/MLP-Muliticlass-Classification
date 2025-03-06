import random
import string
import gensim.downloader as api
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time

# Start timing
start_time = time.time()

np.random.seed(42)

# Load Gensim model (make sure this is done outside of the function to avoid re-loading every time)
model = api.load("glove-wiki-gigaword-50")  # GloVe 50-dimensional model for example

def generate_synonyms_gensim(word, topn = 1):
    """Generate synonyms using Gensim's Word2Vec or GloVe model."""
    try:
        # Get the most similar words to the given word
        similar_words = model.most_similar(word, topn=topn)
        return [synonym for synonym, similarity in similar_words]
    except KeyError:
        return []  # Return empty list if the word is not in the vocabulary

def synonym_replacement(sentence, n=1):
    """Replace n random words in the sentence with their first non-identical synonym."""
    words = sentence.split()
    new_words = words.copy()

    # Identify words with synonyms, removing punctuation and duplicates
    random_word_list = list(set([word.strip(string.punctuation) for word in words if word.strip(string.punctuation) in model]))
    random.shuffle(random_word_list)

    num_replacements = 0
    for random_word in random_word_list:
        synonyms = generate_synonyms_gensim(random_word)
        if synonyms:
            # Replace the word with its first non-identical synonym
            synonym = synonyms[0]
            new_words = [synonym if word.strip(string.punctuation) == random_word else word for word in new_words]
            num_replacements += 1
        if num_replacements >= n:  # Stop after replacing n words
            break

    return " ".join(new_words)

def augment_train_data(train_texts, train_labels):
    # Convert Tensor to list
    train_texts = [text.decode('utf-8') for text in train_texts.numpy()]

    # Convert one-hot encoded labels to indices for easier processing
    train_labels_indices = np.argmax(train_labels, axis=1)

    # Count classes
    unique_classes, class_counts = np.unique(train_labels_indices, return_counts=True)
    max_count = np.max(class_counts)  # Largest class size

    print("\nClass Distribution Before Augmentation:")
    print("Classes:", unique_classes)
    print("Counts:", class_counts)
    print("Max Count:", max_count)

    # Create a list to store the new texts and labels
    new_train_texts = []
    new_train_labels = []

    # Process each class
    for label in unique_classes:
        if class_counts[label] > 0.3 * max_count and class_counts[label] < 0.6 * max_count:
            augment_times = 0 #can vary
        elif class_counts[label] < 0.3 * max_count:
            augment_times = 1
        else:
            augment_times = 0  # No augmentation if class size >= 60% of max count

        # Identify texts belonging to this label
        label_indices = np.where(train_labels_indices == label)[0]
        label_texts = [train_texts[i] for i in label_indices]
        label_one_hot = train_labels[label_indices]

        # Append original texts and labels
        new_train_texts.extend(label_texts)
        new_train_labels.extend(label_one_hot)

        counter = 0
        # Augment texts
        for _ in range(augment_times):
            for text in label_texts:
                counter += 1
                augmented_text = synonym_replacement(text)  # Apply augmentation
                new_train_texts.append(augmented_text)
                new_train_labels.append(label_one_hot[0])  # Append corresponding one-hot label
                if counter % 1000 == 0:
                    print(f"Augmented {counter} texts for class {label}")

    # Shuffle the augmented dataset
    combined = list(zip(new_train_texts, new_train_labels))
    random.shuffle(combined)
    new_train_texts, new_train_labels = zip(*combined)

    # Convert back to TensorFlow Tensors or leave as Python lists, as needed
    new_train_texts = list(new_train_texts)
    new_train_labels = np.array(new_train_labels)

    print("\nClass Distribution After Augmentation:")
    augmented_class_counts= np.unique(np.argmax(new_train_labels, axis=1), return_counts=True)[1]
    print("Counts:", augmented_class_counts)

    return new_train_texts, new_train_labels

# One-hot encoding function
def one_hot_encode_labels(df, num_classes):
    labels = np.zeros((df.shape[0], num_classes))
    for i, label in enumerate(df['label']):
        labels[i, int(label)] = 1
    return labels

def print_class_counts(predictions, num_classes):
    """Print the number of samples predicted for each class."""
    class_counts = np.bincount(predictions, minlength=num_classes)
    print("Class Distribution in Predictions:")
    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i]} samples")

# Activation Functions
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    x = np.clip(x, -1e6, 1e6)
    return x * (1 - x)

def relu(x):
    x = np.clip(x, -1e8, 1e8)
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)  # Returns 1 for x > 0, otherwise 0

def softmax(x):
    x = np.clip(x, -1e6, 1e6)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_derivative(output, label):
    return output - label

class Layer:
    def __init__(self, input_size, output_size, activation, dropout_rate=0.0):
        # Inputs
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Weights and Biases
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.random.randn(1, self.output_size) * 0.1

        # Batch normalization
        self.gamma = np.ones((1, self.output_size))
        self.beta = np.zeros((1, self.output_size))
        self.mean = np.zeros((1, self.output_size))
        self.var = np.ones((1, self.output_size))
        self.momentum = 0.9

        # Dropout
        self.output = None
        self.input_sample = None
        self.mask = None

        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def batch_normalization(self, z, training=True):
        if training:
            batch_mean = np.mean(z, axis=0, keepdims=True)
            batch_var = np.var(z, axis=0, keepdims=True)
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.mean
            batch_var = self.var
        z_norm = (z - batch_mean) / np.sqrt(batch_var + self.epsilon)
        return self.gamma * z_norm + self.beta

    def forward(self, input_sample, training=True):
        self.input_sample = input_sample
        z = np.dot(input_sample, self.weights) + self.biases
        z = self.batch_normalization(z, training)

        if self.activation == 'sigmoid':
            self.output = sigmoid(z)
        elif self.activation == 'relu':
            self.output = relu(z)
        elif self.activation == 'softmax':
            self.output = softmax(z)

        if training and self.dropout_rate > 0.0:
            self.mask = np.random.rand(*self.output.shape) > self.dropout_rate
            self.output *= self.mask
        else:
            self.mask = None

        return self.output

    def backward(self, d_output):
        if self.mask is not None:
            d_output *= self.mask

        if self.activation == 'sigmoid':
            d_output *= sigmoid_derivative(self.output)
        elif self.activation == 'relu':
            d_output *= relu_derivative(self.output)

        d_weights = np.dot(self.input_sample.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_inputs = np.dot(d_output, self.weights.T)

        return d_weights, d_biases, d_inputs

    def update(self, d_weights, d_biases, learning_rate):
        self.t += 1
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)

        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** self.t)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** self.t)

        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)


class FeedForwardNN:
    def __init__(self):
        self.layers = []
        self.history = {"epoch": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    def addLayer(self, input_size, output_size, activation, dropout=0.0):
        layer = Layer(input_size, output_size, activation, dropout)
        self.layers.append(layer)

    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    def evaluate(self, X, y):
        output = self.forward(X, training=False)
        loss = -np.sum(y * np.log(output + 1e-9), axis=1).mean()
        accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size=2048, learning_rate=1e-4):
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                output = self.forward(X_batch)

                losses = -np.sum(y_batch * np.log(output + 1e-9), axis=1)
                loss = np.mean(losses)

                d_output = output - y_batch
                for layer in reversed(self.layers):
                    d_weights, d_biases, d_output = layer.backward(d_output)
                    layer.update(d_weights, d_biases, learning_rate)

             # Compute metrics for this epoch
            train_loss, train_accuracy = self.evaluate(X_train, y_train)
            test_loss, test_accuracy = self.evaluate(X_test, y_test)

            # Save metrics to history
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)


def scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate, epochs, batch_size, set_layers= None, default_layers = True):
    # Initialize the feedforward neural network
    input_size = 50  # Input size (e.g., size of word embedding)
    output_size = num_classes  # Number of output classes

    nn = FeedForwardNN()

    # Layers
    if default_layers:
      nn.addLayer(50, 32, 'relu', dropout = 0.6)  # First hidden layer
      nn.addLayer(32, 16, 'relu', dropout = 0.4)   # Second hidden layer
      nn.addLayer(16, 16, 'relu', dropout = 0.6)   # Third hidden layer
      nn.addLayer(16, 16, 'relu')   # Fourth hidden layer
      nn.addLayer(16, output_size, 'softmax') # Output layer
    else:
      isFirstLayer = True
      for dropout_rate,size,activation in set_layers:
          if isFirstLayer:
            nn.addLayer(50, size, activation)
            previous_layer_size = size
            isFirstLayer = False
          else:
            nn.addLayer(previous_layer_size, size, activation, dropout = dropout_rate)
            previous_layer_size = size

      nn.addLayer(previous_layer_size, output_size, 'softmax')

    # Train the neural network
    nn.train(train_data, train_labels, test_data, test_labels, epochs= epochs, batch_size =batch_size, learning_rate= learning_rate)

    results_df = pd.DataFrame(nn.history)

    # Evaluate the model
    train_preds = nn.predict(train_data)
    train_accuracy = np.mean(train_preds == np.argmax(train_labels, axis=1))
    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

    test_preds = nn.predict(test_data)

    test_accuracy = np.mean(test_preds == np.argmax(test_labels, axis=1))
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    print("Training Set Predictions:")
    print_class_counts(train_preds, num_classes)

    print("\nTest Set Predictions:")
    print_class_counts(test_preds, num_classes)

    return results_df

# Read the dataset file
dataframe = pd.read_csv('dataset1.csv')
# sad(0), joy(1), love(2), anger(3), fear(4), surprise(5)
num_classes = 6
df = dataframe.dropna()

# Split the data
train, test = np.split(df.sample(frac=1), [int(0.6 * len(df))])
train_labels = one_hot_encode_labels(train, num_classes)
test_labels = one_hot_encode_labels(test, num_classes)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

train_text = tf.convert_to_tensor(train['text'].values, dtype=tf.string)
test_text = tf.convert_to_tensor(test['text'].values, dtype=tf.string)

train_text, train_labels = augment_train_data(train_text, train_labels)

# Get the embeddings for the text
train_data = hub_layer(train_text).numpy()
test_data = hub_layer(test_text).numpy()

# Normalize data
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)

save_default_results = False
save_some_layers_results = False
save_many_layers_results = False
save_many_many_layers_results = True
save_diff_learning_rate = False
save_diff_batch_size = False

#Default values
learning_rate = 0.0001
epochs = 30
batch_size = 2048

#scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate, epochs, batch_size, set_layers= None, default_layers = True)

if save_default_results:
    # Run the scratch model
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)

    #Results for default values
    scr_result.to_csv('scr_default_param.csv', index=False)

if save_some_layers_results:
    #set layers -> dropout_rate,size,activation
    #set some layers with small size and some dropout
    set_layers_1 = [(0.0, 16, 'relu'),(0.2, 16, 'relu'),(0.6, 16, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    scr_result.to_csv('scr_some_layers_1.csv', index=False)
    #set some layers with bigger size and some dropout
    set_layers_2 = [(0.0, 32, 'relu'),(0.6, 64, 'relu'),(0.4, 32, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    scr_result.to_csv('scr_some_layers_2.csv', index=False)


if save_many_layers_results:
    #set layers -> isDropout, size, activation
    #set some more layers with small size and some dropout
    set_layers_1 = [(0.0, 16, 'relu'),(0.2, 16, 'relu'),(0.6, 16, 'relu'),(0.0, 16, 'relu'),(0.4, 16, 'relu'),(0.2, 16, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    scr_result.to_csv('scr_many_layers_1.csv', index=False)
    #set some more layers with bigger size and some dropout
    set_layers_2 =  [(0.0, 128, 'relu'),(0.4, 64, 'relu'), (0.6 , 32 , 'relu'),(0.0, 32, 'relu'),(0.4, 16, 'relu'),(0.2, 16, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    scr_result.to_csv('scr_many_layers_2.csv', index=False)

if save_many_many_layers_results:
    #set layers -> isDropout, size, activation
    #set some more layers with small size and some dropout
    set_layers_1 = [(0.0, 16, 'relu'),(0.2, 16, 'relu'),(0.6, 16, 'relu'),(0.0, 16, 'relu'),(0.4, 16, 'relu'),(0.2, 16, 'relu'),(0.0, 16, 'relu'),(0.1, 16, 'relu'),(0.3, 16, 'relu'),(0.0, 16, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    scr_result.to_csv('scr_many_many_layers_1.csv', index=False)
    #set some more layers with bigger size and some dropout
    set_layers_2 =  [(0.0, 128, 'relu'),(0.4, 64, 'relu'), (0.6 , 64 , 'relu'),(0.0, 32, 'relu'),(0.4, 32, 'relu'),(0.2, 16, 'relu'),(0.1, 16, 'relu'),(0.2, 16, 'relu')]
    scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    scr_result.to_csv('scr_many_many_layers_2.csv', index=False)

if save_diff_learning_rate:
    learning_rates = [0.01, 0.001, 0.0001]
    for learning_rate in learning_rates:
        scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)
        scr_result.to_csv('scr_lr_' + str(learning_rate) + '.csv', index=False)

if save_diff_batch_size:
    batch_sizes = [2048, 4096, 8192]
    for batch_size in batch_sizes:
        scr_result = scratch_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)
        scr_result.to_csv('scr_bs_' + str(batch_size) + '.csv', index=False)

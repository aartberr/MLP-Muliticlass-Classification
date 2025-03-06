#nearest centroid algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

np.random.seed(42)

def calculate_class_centroid(num_classes,train):
    train_vectors = np.vstack(train['text_embeddings'].to_numpy())
    train_labels = train['text_label'].to_numpy()

    # Initialize an empty dictionary to classify the train samples
    indices_dict = {}

    for label in range(num_classes):
      indices_dict[label] = np.where(train_labels == label)[0]

    # Calculate the mean of coordinates of all vectors of each class
    class_centroids = {}
    for label, label_indices in indices_dict.items():
      class_centroids[label] = []
      class_centroids[label].append(train_vectors[label_indices].mean(axis=0))

    return class_centroids # Return centroid vector dictionary {label: centroid_vector}

def nearest_centroid_accuracy(num_classes, test, class_centroids):
    accuracy_counter = 0
    centroid_vectors_list = []
    centroid_labels_list = []

    for label in range(num_classes):
      centroid_label, centroid_vector = list(class_centroids.items())[label]
      centroid_vectors_list.append(centroid_vector)
      centroid_labels_list.append(centroid_label)

    centroid_vectors = np.array(centroid_vectors_list)
    centroid_labels = np.array(centroid_labels_list)

    centroid_vectors = np.squeeze(centroid_vectors)

    for test_vector, test_label in zip(test['text_embeddings'], test['text_label']):
      distances = []
      test_vector = test_vector.reshape(1, -1)

      euclidean_distances = np.linalg.norm(centroid_vectors - test_vector, axis=1)
      distances = list(zip(euclidean_distances, centroid_labels))

      distances.sort(key=lambda x: x[0])
      closest_centroid_labels = distances[0][1]

      if test_label == closest_centroid_labels:
        accuracy_counter +=1

    return accuracy_counter



# Read the dataset file
dataframe = pd.read_csv('dataset1.csv')
# sad(0), joy(1), love(2), anger(3), fear(4), surprise(5)
num_classes = 6
df = dataframe

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

# Convert text to tensor
text_tensor = tf.convert_to_tensor(df['text'].tolist())

# Get the embeddings
embeddings = hub_layer(text_tensor)

# Convert embeddings to a NumPy array if needed
embeddings_np = embeddings.numpy()

# Split the data
data_to_pd_dataframe = {
    'text_embeddings': list(embeddings_np),
    'text_label': df['label']
}

data_to_split = pd.DataFrame(data_to_pd_dataframe, columns=['text_embeddings', 'text_label'])

train, test = np.split(data_to_split.sample(frac=1), [int(0.6 * len(data_to_split))])

class_centroids = calculate_class_centroid(num_classes, train)

accuracy_counter = nearest_centroid_accuracy(num_classes, test, class_centroids)

print(accuracy_counter/len(test))
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def voting(k, distances, num_classes):
    classes_votes = np.zeros((1, num_classes))
    for i in range(k):
        label = distances[i][1]
        classes_votes[0, int(label)] += 1
    return classes_votes.flatten()

def k_nearest_neighbor(test_vector, test_label, train_vectors, train_labels, k, num_classes):
    # Compute Euclidean distances between test vector and all training vectors
    euclidean_distances = np.linalg.norm(train_vectors - test_vector, axis=1)
    distances = list(zip(euclidean_distances, train_labels))

    # Take the k closest neighbors
    distances.sort(key=lambda x: x[0])
    classes_votes = voting(k, distances, num_classes)

    # Find if there is one or more most-voted classes
    index = k
    while True:
        max_value = classes_votes.max()
        max_indices = np.where(classes_votes == max_value)[0]
        if len(max_indices) == 1:  # Only one class with the most votes
            if max_indices[0] == test_label:
                return 1  # Correct prediction
            else:
                return 0  # Incorrect prediction
        else:
            index -= 1
            classes_votes = voting(index, distances, num_classes)

def k_nearest_neighbor_accuracy(k, num_classes, test, train, batch_size=1000):
    accuracy_counter = 0
    num_batches = len(test) // batch_size + (1 if len(test) % batch_size != 0 else 0)

    # Transform train embeddings and their labels to numpy arrays
    train_vectors = np.vstack(train['text_embeddings'].to_numpy())
    train_labels = train['text_label'].to_numpy()

    # Limit the number of workers to some reasonable value
    max_workers = 6

    # Iterate over the batches and parallelize the processing of each batch
    for batch_idx in range(num_batches):
        futures = []
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(test))

        batch_test = test.iloc[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each test vector in the current batch
            for test_vector, test_label in zip(batch_test['text_embeddings'], batch_test['text_label']):
                futures.append(executor.submit(k_nearest_neighbor, test_vector, test_label, train_vectors, train_labels, k, num_classes))

            # Track completed futures and update accuracy
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Batch {batch_idx + 1}/{num_batches}'):
                accuracy_counter += future.result()


    return accuracy_counter

# Set the number of nearest neighbors
k = 3

# Read the dataset file
dataframe = pd.read_csv('dataset1.csv')
# sad(0), joy(1), love(2), anger(3), fear(4), surprise(5)
num_classes = 6
testing_nn_sample = 10000
df = dataframe.head(testing_nn_sample)

# Load the pre-trained embedding model from TensorFlow Hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

# Convert text to tensor and get embeddings
text_tensor = tf.convert_to_tensor(df['text'].tolist())
embeddings = hub_layer(text_tensor)

# Convert embeddings to a NumPy array
embeddings_np = embeddings.numpy()

# Prepare the data for splitting
data_to_pd_dataframe = {
    'text_embeddings': list(embeddings_np),
    'text_label': df['label']
}

data_to_split = pd.DataFrame(data_to_pd_dataframe, columns=['text_embeddings', 'text_label'])

# Split the data into train and test sets (60% training, 40% testing)
train, test = np.split(data_to_split.sample(frac=1), [int(0.6 * len(data_to_split))])

# Calculate accuracy using parallelized k-NN
accuracy_counter = k_nearest_neighbor_accuracy(k, num_classes, test, train)

# Output the accuracy
print(f"Accuracy: {accuracy_counter}-{len(test)}  {accuracy_counter / len(test)}")

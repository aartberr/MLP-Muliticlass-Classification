import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

np.random.seed(42)

def one_hot_encode_labels(df, num_classes):
  # Use one-hot encoding
  labels = np.zeros((df.shape[0], num_classes))
  # Set the label for each row
  for i, label in enumerate(df['label']):
    labels[i, int(label)] = 1

  return labels

# Format and batch the data
def df_to_dataset(dataframe, labels, batch_size = 1024):
    df = dataframe.copy()
    df = df['text']
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate, epochs, batch_size, set_layers= None, default_layers = True):
    train_data = df_to_dataset(train, train_labels, batch_size)
    test_data = df_to_dataset(test, test_labels, batch_size)

    # Model
    model = tf_keras.Sequential()

    # Embedding Layer
    hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

    # Layers
    if default_layers:
        model.add(hub_layer) #text -> vector
        model.add(tf_keras.layers.Dense(32, activation = 'relu')) # First hidden layer
        model.add(tf_keras.layers.Dropout(0.6))
        model.add(tf_keras.layers.Dense(16, activation = 'relu')) # Second hidden layer
        model.add(tf_keras.layers.Dropout(0.4))
        model.add(tf_keras.layers.Dense(16, activation = 'relu')) # Third hidden layer
        model.add(tf_keras.layers.Dropout(0.6))
        model.add(tf_keras.layers.Dense(16, activation = 'relu')) # Fourth hidden layer
    else:
      model.add(hub_layer) #text -> vector
      for isDropout,size,activation in set_layers:
          if isDropout:
            model.add(tf_keras.layers.Dropout(size))
          else:
            model.add(tf_keras.layers.Dense(size, activation = activation))

    # Output Layer
    model.add(tf_keras.layers.Dense(num_classes, activation = 'softmax'))

    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate = learning_rate), loss=tf_keras.losses.CategoricalCrossentropy(),metrics = ['accuracy'])

    history = model.fit(train_data, epochs = epochs, validation_data = test_data)

    # Collect results
    results = {
        "epoch": list(range(1, epochs + 1)),
        "train_loss": history.history['loss'],
        "train_accuracy": history.history['accuracy'],
        "test_loss": history.history['val_loss'],
        "test_accuracy": history.history['val_accuracy'],
    }

    results_df = pd.DataFrame(results)

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

save_default_results = False
save_some_layers_results = False
save_many_layers_results = False
save_many_many_layers_results = True
save_diff_learning_rate = False
save_diff_batch_size = False

#Default values
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
learning_rate = 0.001
epochs = 30
batch_size = 2048

#libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate, epochs, batch_size)

if save_default_results:
    # Run the libraries based model
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)

    #Results for default values
    lib_result.to_csv('lib_default_param.csv', index=False)

if save_some_layers_results:
    #set layers -> isDropout, size, activation
    #set some layers with small size and some dropout
    set_layers_1 = [(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.2, None), (False, 16, 'relu'),(True, 0.6, None)]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    lib_result.to_csv('lib_some_layers_1.csv', index=False)
    #set some layers with bigger size and some dropout
    set_layers_2 = [(False, 32, 'relu'),(False, 64, 'relu'),(True, 0.6, None), (False, 32, 'relu'),(True, 0.4, None)]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    lib_result.to_csv('lib_some_layers_2.csv', index=False)


if save_many_layers_results:
    #set layers -> isDropout, size, activation
    #set some more layers with small size and some dropout
    set_layers_1 = [(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.2, None), (False, 16, 'relu'),(True, 0.6, None),(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.4, None),(False, 16, 'relu'),(True, 0.2, None)]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    lib_result.to_csv('lib_many_layers_1.csv', index=False)
    #set some more layers with bigger size and some dropout
    set_layers_2 =  [(False, 128, 'relu'),(False, 64, 'relu'),(True, 0.4, None), (False,32 , 'relu'),(True, 0.6, None),(False, 32, 'relu'),(False, 16, 'relu'),(True, 0.4, None),(False, 16, 'relu'),(True, 0.2, None)]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    lib_result.to_csv('lib_many_layers_2.csv', index=False)

if save_many_many_layers_results:
    #set layers -> isDropout, size, activation
    #set some more layers with small size and some dropout
    set_layers_1 = [(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.2, None), (False, 16, 'relu'),(True, 0.6, None),(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.4, None),(False, 16, 'relu'),(True, 0.2, None),(False, 16, 'relu'),(False, 16, 'relu'),(True, 0.1, None),(False, 16, 'relu'),(True, 0.3, None),(False, 16, 'relu')]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_1, default_layers = False)
    #Results
    lib_result.to_csv('lib_many_many_layers_1.csv', index=False)
    #set some more layers with bigger size and some dropout
    set_layers_2 =  [(False, 128, 'relu'),(False, 64, 'relu'),(True, 0.4, None), (False,64 , 'relu'),(True, 0.6, None),(False, 32, 'relu'),(False, 32, 'relu'),(True, 0.4, None),(False, 16, 'relu'),(True, 0.2, None),(False, 16, 'relu'),(True, 0.1, None),(False, 16, 'relu'),(True, 0.2, None)]
    lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, set_layers = set_layers_2, default_layers = False)
    #Results
    lib_result.to_csv('lib_many_many_layers_2.csv', index=False)

if save_diff_learning_rate:
    learning_rates = [0.01, 0.001, 0.0001]
    for learning_rate in learning_rates:
        lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)
        lib_result.to_csv('lib_lr_' + str(learning_rate) + '.csv', index=False)

if save_diff_batch_size:
    batch_sizes = [2048, 4096, 8192]
    for batch_size in batch_sizes:
        lib_result = libraries_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size)
        lib_result.to_csv('lib_bs_' + str(batch_size) + '.csv', index=False)

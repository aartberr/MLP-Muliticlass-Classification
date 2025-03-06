# MLP Multiclass Classification with Comparison to KNN and NCC  

This project was created as part of a course assignment for the **Neural Networks** class at **CSD, AUTH University**.

This project explores **multiclass text classification** using **neural networks**, as well as traditional classification methods like **K-Nearest Neighbors (KNN)** and **Nearest Class Centroid (NCC)**.  

## **Implemented Methods**  

1. **K-Nearest Neighbors (KNN) Classifier**  
   - Includes a **parallelized approach** for faster computation.  
   - Implemented in `k_nearest_neighbor.py`.  

2. **Nearest Class Centroid (NCC) Classifier**  
   - A simple centroid-based classification method.  
   - Implemented in `nearest_class_centroid.py`.  

3. **Multilayer Perceptron (MLP) - Library-Based Model**  
   - Uses `tf.keras` to build and train a neural network.  
   - Supports multiple **configurable hyperparameters** (e.g., number of layers, layer size, dropout rate, learning rate, batch size).  
   - Can run multiple experiments with different parameters and plot results.  
   - Implemented in `library_based_model.py`.  

4. **MLP - Custom Implementation**  
  - A manually implemented neural network class that follows the same structure as the library-based model.  
   - Shares the same hyperparameters for direct comparison.  
   - Implemented in `scratch_model.py`.  

## **Dataset Requirements**  

The models are designed to work with any **text dataset** that maps texts to **numerical classes (not ordinal)**. The dataset should be in CSV format and named:  

```
dataset1.csv
```

## **Key Features**  

- The accuracy and loss results per epoch can be saved in 2 `.csv` files by enabling the corresponding flag.  
  - Example: `save_diff_learning_rate = True` saves accuracy results for different learning rates.  
- The comparison between the **scratch MLP model** and the **library-based model** provides insights into:  
  - How an MLP works internally.  
  - The challenges of manually implementing an efficient MLP.  
  - The performance gap between an optimized framework-based model and a custom-built one. 

## **Files**  

- **k_nearest_neighbor.py** – Implements the KNN classifier with a parallelized approach.  
- **nearest_class_centroid.py** – Implements the NCC classifier.  
- **library_based_model.py** – Implements an MLP classifier using `tf.keras`, allowing parameter tuning.  
- **scratch_model.py** –  Implements a manually built MLP for comparison.  
- **dataset1.csv** – The dataset file (should be provided by the user).  

## **Usage**  

1. **Run KNN or NCC classification**:  
   ```bash
   python k_nearest_neighbor.py
   python nearest_class_centroid.py
   ```
  
2. **Train the MLP using `tf.keras`**:  
   ```bash
   python library_based_model.py
   ```

3. **Train the MLP from scratch**:  
   ```bash
   python scratch_model.py
   ```

## **Future Work**  

- Improve the scratch MLP implementation with additional optimizations.  
- Extend to other classification tasks, such as image classification.  

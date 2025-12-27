# CENG463_HW_IMG_CLUSTERING

# CENG 463 – HW3: CIFAR-10 Image Clustering

As an **image clustering task**, you will work with the CIFAR-10 dataset, which contains images from 10 classes, including airplanes, cars, cats, and dogs. The goal is to use **unsupervised learning** techniques to group similar images automatically and visualize the results.

## 1. Dataset Overview 
- Load the dataset using Keras (More information: [Keras CIFAR-10.](https://keras.io/api/datasets/cifar10/)):

    from keras.datasets import cifar10

   (x_train, y_train), (x_test, y_test) = cifar10.load_data()

- Show the dataset size and class names in a table.
- Display sample images from each class.

## 2. Feature Extraction and PCA 
- Convert each image from 32×32×3 into a 3072-dimensional vector.
- Scale the vectors using StandardScaler.
- Apply PCA to reduce the dimensionality from 3072 → 50.

## 3. Clustering Methods 
- Perform clustering on the PCA-transformed features using three Sklearn algorithms: K-Means, Agglomerative Clustering, DBSCAN.
- Visualize example images for each cluster.
- Provide short comments on the clusters.

## 4. Visualization and Analysis (30 pts)
- Use t-SNE to reduce PCA features to 2D for visualization.
- Create colored scatter plots for each clustering algorithm.
- Observe which classes are better grouped by each algorithm.
 

# KNN and K-Means README
 **Project**: KNN + K-Means<br>
 **Class**: Northwestern CS 349 Fall 2024<br>
 **Contributers**: Raymond Gu, Mimi Zhang, Alvin Xu, Rhema Phiri, Eshan Haq

## Handwriting Datasets
The different `mnist` datasets are handwriting datasets. The dataset contains images of handwritten
numbers and their corresponding labels. Each image provided has a 28x28 pixel dimension.

## Starter File
The `starter.py` file contains all the functions needed for KNN and K-Means.<br>

**Description For Preprocessing Function**<br>
The `preprocess` function makes several adjustments to the dataset that make it easier to work with.
- **Grayscale to Binary**: It converts all the pixels from grayscale (range from 0 - 255) to binary with a cutoff of 128.<br>
- **Cropping**: The original image is 28x28 pixels. After cropping the image, it is 22x24 pixels.<br><br>

**Description For Updated_Show Function**<br>
The `updated_show` function shows what the data looks like after being preprocessed.<br>
- **Note**: This function only works for 22x24 images, it does not work for the original images.<br><br>

**Description For KNN Function**<br>
The `knn` function returns a list of labels for the query dataset based upon labeled observations in the train dataset.

- **Number of Neighbors**: The number of neighbors is hard-coded at 4.<br>
- **Distance Metric**: The metric used by the algorithm can either be Euclidean or Cosine Similarity. The user can choose
                       which metric they want the algorithm to use.<br><br>

**Description For K-Means Function**<br>
The `kmeans` function returns a list of labels for the query dataset based upon observations in the train dataset.
- **Number of Clusters**: The number of clusters is set to 20.<br>
- **Number of Neighbors**: When classifying an example, the function uses the 4 closest centroids to make a prediction.<br>
- **Distance Metric**: The metric used by the algorithm can either be Euclidean or Cosine Similarity. The user can choose
                       which metric they want the algorithm to use.<br><br>

## Tester File
The `tester.py` file allows the user to see the following results.
- The accuracy of the `knn` function on the validation and test datasets.
- The accuracy and f1-score of the `kmeans` function on the validation and test datasets.

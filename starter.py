# Class: Northwestern CS 349 Fall 2024
# ------------------------------------

# Professor: David Demeter
# ------------------------------------

# Contributers:
# ------------------------------------
#   Raymond Gu
#   Mimi Zhang
#   Alvin Xu
#   Rhema Phiri
#   Eshan Haq

import math
import random
import heapq

# Euclidean Function
# **************************************************************************
'''
 Purpose: Returns Euclidean distance given vectors and b

 Note:
 - Since the data is transformed from greyscale to binary, euclidean
   will be less accurate than other distance metrics.
'''
def euclidean(a,b):
    if len(a) != len(b):
        raise ValueError("Error: Vector lengths are not equal!")
    else: 
        sum_radicand = 0
        for i in range(len(a)):
             radicand = (a[i]- b[i]) ** 2
             sum_radicand+= radicand 
        dist = math.sqrt(sum_radicand)
    return(dist)

# Cosim Function
# **************************************************************************
'''
 Purpose: Returns Cosine Similarity given vectors and b

 Note:
 - This function returns Cos(theta). This is a similarity score NOT
   a distance metric. To turn this into a distance metric, we want to
   do 1 - Cos(theta) (when theta = 0, distance = 0).
'''
def cosim(a,b):
   if len(a) != len(b):
        raise ValueError("Error: Vector lengths are not equal!")
   else: 
    size = len(a)
    accum_mag_a = 0
    accum_mag_b = 0
    accum_dot_product = 0

    for i in range(size):
       mag_a = (a[i]- 0) ** 2
       mag_b = (b[i]- 0) ** 2
       accum_mag_a += mag_a
       accum_mag_b += mag_b
       dot_product = a[i] * b[i]
       accum_dot_product += dot_product

    total_mag_a = math.sqrt(accum_mag_a)
    total_mag_b = math.sqrt(accum_mag_b)

    if (total_mag_a * total_mag_b == 0):
     raise ValueError("Error: Division by 0")
   
    else:
     dist  = accum_dot_product / (total_mag_a * total_mag_b)

   return(dist)

# KNN Function
# **************************************************************************
def knn(train,query,metric):
    '''
    Purpose: Returns a list of labels for the query dataset based upon labeled
            observations in the train dataset.

    Notes:
    - Metric is a string specifying either "euclidean" or "cosim".
    - All hyper-parameters (value of K) should be hard-coded in the algorithm.
    '''
    # Initialize list of predicted labels for the query dataset
    labels = []

    # Define number of neighbors being used (k value)
    k = min(4, len(train))
    
    # Loop through the query dataset
    for q in query:

        # Make a priority queue that stores distance-label pairs
        distances = []

        # Get the pixels of the query dataset (ignore the labels)
        q_label = q[0]
        q_pixels = list(map(int, q[1])) # Convert string to int

        # Loop through the training dataset
        for t in train:
            # Get the label and the pixels of the training dataset
            t_label = t[0]
            t_pixels = list(map(int, t[1])) # Convert string to int

            # Calculate the distance from each of the datapoints
            if metric == 'euclidean':
                dist = euclidean(q_pixels, t_pixels)
            if metric == 'cosim':
                dist = 1 - cosim(q_pixels, t_pixels)

            # Add the distance from the datapoint to the priority queue
            heapq.heappush(distances, (dist, t_label))
        
        # Make dictionary that stores {label: {count, avg_dist}}
        k_nearest = {}

        # Get the k nearest neighbors and add them to the dictionary
        for i in range(k):
            
            # Pop the distance and label from the next neighbor
            dist, key = heapq.heappop(distances)

            # If the label is in the dictionary, add 1 to count and update avg_dist
            if key in k_nearest:
               entry = k_nearest[key]
               entry["avg_dist"] = (entry["avg_dist"] * entry["count"] + dist) / (entry["count"] + 1)
               entry["count"] = entry["count"] + 1

            # Otherwise, just add the label to the dictionary
            else:
               k_nearest[key] = {"count":1, "avg_dist":dist}
        
        # Get the label with the most appearances
        max_count = 0
        closest_dist = math.inf
        result = None
        for key, value in k_nearest.items():
            count = value["count"]
            avg_dist = value["avg_dist"]
            if count > max_count:
                max_count = count
                closest_dist = avg_dist
                result = key
            elif count == max_count:
               if avg_dist < closest_dist:
                   closest_dist = avg_dist
                   result = key

        # Add the label with the most appearance to the list
        labels.append(result)
    
    # Return the predicted labels
    return (labels)
                
# Test_KNN Function
# **************************************************************************
'''
 Purpose: Tests the KNN function against the test or validation set
'''
def test_knn(query,metric):

    # Make sure a valid input was provided
    if (query=="Train"):
        path = "mnist_train.csv"
    elif (query == "Test"):
        path = "mnist_test.csv"
    elif (query == "Validation"):
        path = "mnist_valid.csv"
    else:
        print("    Error: Did not provide valid input")
        return

    # Get the training and query data
    query_data = preprocess(path)
    training_data = preprocess("mnist_train.csv")

    # Get the predictions
    predictions = knn(training_data, query_data, metric)

    # Get the number of correct predictions
    correct_predictions = 0
    total_num_query = len(query_data)
    for i in range(total_num_query):
        if (query_data[i][0] == predictions[i]):
            correct_predictions += 1
    
    # Calculate and return the accuracy
    accuracy = correct_predictions / total_num_query * 100
    return accuracy

# Kmeans Function
# **************************************************************************
def kmeans(train,query,metric):
    '''
    Purpose: Returns a list of labels for the query dataset based upon
             observations in the train dataset. 

    Notes:
    - Labels should be ignored in the training set
    - Metric is a string specifying either "euclidean" or "cosim".  
    - All hyper-parameters (value of K) should be hard-coded in the algorithm.
    '''
    # Initialize variables to store best clustering and the total variance
    best_clustering = None
    lowest_variance = math.inf

    # Randomly pick 10 data points to be centroids
    random_data = random.sample(train, 20)

    # Make list to store centroids
    centroids = []

    # Manually put each data point into the dictionary
    for i in range(20):

        # Give the centroid a label
        centroid_label = "Cluster-" + str(i)

        # Get the pixel data for the centroid
        centroid_pixels = random_data[i][1]

        # Add the centroid to the list
        centroid = [centroid_label, centroid_pixels]
        centroids.append(centroid)

    # Calculate the final centroids after making the clusters
    centroid_cluster = get_final_centroids(centroids, train, metric, 0)
    
    # Give labels to each of the centroids
    final_centroids = []
    for cluster in centroid_cluster:
        cluster_label = label_cluster(cluster[1])
        cluster_center = cluster[0][1]

        final_centroids.append([cluster_label, cluster_center])

    # Use KNN to return the label of the query data
    return knn(final_centroids, query, metric)

# Make Cluster Function
# **************************************************************************
def get_final_centroids(centroids, train, metric, k):
    '''
    Purpose: Make the clusters given random centroids and training data.

    Notes:
    - Should return the a list of the centroid-cluster pairs
    '''
    # Use KNN to train the data on the current centroids
    labels = knn(centroids, train, metric)

    # Create clusters {label: [x_1, x_2, x_3, ...]}
    clusters = {}

    # Loop through data points and put them in their clusters
    for i, q in enumerate(train):

        # Add data point to it's cluster
        if labels[i] in clusters: clusters[labels[i]].append(q)
        else: clusters[labels[i]] = [q]

    # Make variable to track if centroids have been updated
    updated = False

    # Create list to store centroid & cluster pairs
    centroid_cluster = []

    # Loop through clusters
    for i, cluster in enumerate(clusters.values()):

        # Calculate new centroid of centroid & update if needed
        middle = calculate_middle(cluster)

        if centroids[i][1] != middle:
            centroids[i][1] = middle
            updated = True
        else:
            centroid_cluster.append([centroids[i], cluster])
            continue

    if not updated:
        # Have to return centroids with adjusted labels
        return centroid_cluster
    else:
        return get_final_centroids(centroids, train, metric, k+1)

# Calculate_Middle Function
# **************************************************************************
def calculate_middle(cluster):
    '''
    Purpose: Calculate the new center of a cluster.

    Notes:
    - Cluster is a list of examples [example_1, example_2, ...]
    - Each example has a label and pixel data [label, [p1, p2, ...]]
    - The pixel data consists of strings NOT integers
    '''
    # Initialize the center to return
    center = []
    for i in range(len(cluster[0][1])) : center.append(0)

    # Loop through each example in the cluster
    for example in cluster:     
        # Get the pixel data for the example
        pixel_data = list(map(int, example[1]))

        # Loop through the pixel_data
        for i in range(len(pixel_data)):
            center[i] = center[i] + pixel_data[i]

    # Get the average of each dimension
    num_examples = len(cluster)
    for i in range(len(center)):

        new_middle = center[i] / num_examples
        if new_middle >= 0.5:
            center[i] = 1
        else:
            center[i] = 0

    # Return the new center
    return list(map(str, center))

# Label Clusters Function
# **************************************************************************
def label_cluster(cluster):
    '''
    Purpose: Returns the label that occurs most in a cluster.
    '''
    labels = {}

    # Get all the labels in the cluster
    for example in cluster:
        label = example[0]
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

    # Find the label that occcurs the most
    for label in labels:
        if labels[label] == max(labels.values()):
            return label
            
# F-Measure Function
# **************************************************************************
'''
 Purpose: Returns a an avg f-measure value given a matrix 

 Notes:
 - Precision, recall, and f-measure are set to 0 as default 
 - If there is a lack of clustering for a value, the f_measure is worse
'''
def f_measure(matrix):
    tot_f_measure = 0 
    for row_num in range(len(matrix)):
        tot_f_measure += f_measure_helper(row_num, matrix)
    
    return tot_f_measure / len(matrix)

# F-Measure Helper Function
# **************************************************************************
'''
 Purpose: Returns the f_measure for the real_label value passed in 

 Notes:
 - Precision, recall, and f-measure are set to 0 as default
'''
def f_measure_helper(real_label, matrix):
    # Get true positive value
    tp = matrix[real_label][real_label]

    # Get false positive value
    fp = 0
    for row in range(len(matrix)):
        fp += matrix[row][real_label]

    fp = fp - tp

    # Get false negative value
    fn = 0
    for col in range(len(matrix[real_label])):
        fn += matrix[real_label][col]
        
    fn = fn - tp

    precision = 0
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    
    recall = 0
    if (tp + fn) != 0:
        recall = tp / (tp + fn)

    f_measure = 0
    if ((precision + recall) != 0):
        f_measure = 2 * ((precision * recall) / (precision + recall))
        
    return f_measure

# Preprocess Function
# **************************************************************************
def preprocess(file_name):
    '''
    Purpose: Does preprocessing on each of the examples after reading.

    Notes:
    - We convert from greyscale to binary
    - We crop the image from 28 x 28 to 22 x 24 for dimension reduction
    ''' 
    # Get the original data
    original_data = read_data(file_name)
    
    # Convert the dataset from grayscale to binary
    binary_dataset = grayscale_to_binary(original_data)

    # Crop the image vertically to 22 x 28 dimensions
    cropped_dataset_v = crop_vertical(binary_dataset)

    # Crop the image horizontally to 22 x 24 dimensions
    cropped_dataset_h = crop_horizontal(cropped_dataset_v)

    # Return the preprocessed data
    return(cropped_dataset_h)

# Read_data
# **************************************************************************
def read_data(file_name):
    '''
    Purpose: This function converts the file's contents into a structured dataset.
    '''
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])

    return(data_set)

# Greyscale_to_binary
# **************************************************************************
def grayscale_to_binary(examples):
    '''
    Purpose: Reduce the range of values for possible attributes.
             Right now, the value of each pixel can range from 0 - 255.
             This function will reduce the values to either 0 or 1 (128 cutoff).

    Notes:
    - Examples are in the format [label, [pixel_1, pixel_2, ...]]
    - The labels and attributes of each example are strings
    '''
    # Make a list of updated examples to return
    updated_examples = []

    # Loop through the list of possible examples.
    for example in examples:

        # Get the label of the example and the attributes
        example_label = example[0]
        example_attributes = example[1]

        # Create a list of updated attributes
        updated_attributes = []

        # Loop through the attributes and convert to binary
        for attribute in example_attributes:

            if int(attribute) >= 128:
                updated_attributes.append("1")
            else:
                updated_attributes.append("0")
        
        # Add the updated example to the list
        updated_examples.append([example_label, updated_attributes])

    # Return the list of updated examples
    return updated_examples

# Crop_vertical
# **************************************************************************
def crop_vertical(examples):
    '''
    Purpose: Crop the image vertically to remove rows that don't have
             any pixels. This function will crop 3 rows from the top 
             and 3 from the bottom.

    Notes:
    - Each image is currently 28 x 28 (it will be 22 x 28 after)
    '''
    # Create a list of updated examples
    updated_examples = []

    # Loop through the list of examples
    for example in examples:

        # Get the label and attributes
        label = example[0]
        attributes = example[1]

        # Remove 3 rows from the top and bottom
        row_4_start = 3 * 28
        row_25_end = 25 * 28
        cropped_image = attributes[row_4_start:row_25_end]

        # Add the updated example back to the list
        updated_example = [label, cropped_image]
        updated_examples.append(updated_example)

    # Return the list of updated examples
    return updated_examples

# Crop_horizontal
# **************************************************************************
def crop_horizontal(examples):
    '''
    Purpose: Crop the image horizontally to remove columns that don't have
             any pixels. This function will crop 4 columns from the left

    Notes:
    - Each image is currently 22 x 28 (it will be 22 x 24 after)
    '''
    # Create a list of updated examples
    updated_examples = []

    # Create variables for crop amount and row length
    row_pixels = 28
    crop_amount = 4

    # Loop through the list of examples
    for example in examples:

        # Get the label and attributes
        label = example[0]
        attributes = example[1]

        # Create list to store pixels kept in image
        cropped_image = []

        # Loop through each row and crop 4 pixels
        for row in range(22):

            # Crop 4 pixels from the row
            row_start = row_pixels * row + crop_amount
            row_end = row_pixels * (row+1)
            cropped_row = attributes[row_start:row_end]

            # Add the cropped row to the list of pixels kept
            cropped_image.extend(cropped_row)

        # Add the updated example back to the list
        updated_example = [label, cropped_image]
        updated_examples.append(updated_example)

    # Return the list of updated examples
    return updated_examples

# Updated_Show Function
# **************************************************************************
def updated_show(file_name, mode):
    '''
    Purpose: A new show function to help show the image after preprocessing.
             We reduce the dimensions of the image, so we need a new function
             to display it properly.

    Notes:
    - Old show function only works for 28 x 28 images
    - This function works for 22 x 24 images
    '''
    data_set = preprocess(file_name)

    for obs in range(len(data_set)):
        for idx in range(528):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 24) == 23:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

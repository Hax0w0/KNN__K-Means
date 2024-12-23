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

import starter

def main():
   print()
   print("Welcome To KNN and K-Means Testing")
   print("**********************************************************")

   print("\n   To check accuracy of KNN, type 'knn'")
   print("   To check accuracy and f1-score of K-Means, type 'kmeans'\n")

   response = input("      Please input a method:")
   print()

   if (response == 'knn'):
      test_knn()
   elif (response == 'kmeans'):
      test_kmeans()
   else:
      print("   Invalid input provided")
      return


def test_knn():

   # Print directions
   print("Checking Accuracy of KNN Function")
   print("**********************************************************")
   print("\n   For euclidean, please input e")
   print("   For cosine similarity, please input c\n")

   # Get the distance measure from the tester
   metric = input("      Please enter a distance metric: ")

   # Print directions
   print("\n   To check validation set accuracy, input v")
   print("   To check test set accuracy, input t\n")

   query = input("      Please enter a set to check: ")

   # Write execution for euclidean distance metric
   if (metric == 'e'):
       
      if (query == 't'):
       # Should take less than 1:10
       test_accuracy = starter.test_knn("Test", "euclidean")
       print(f"\n   Test Set Accuracy: {test_accuracy}%")

      elif (query == 'v'):
       # Should take less than 1:10
       valid_accuracy = starter.test_knn("Validation", "euclidean")
       print(f"\n   Validation Set Accuracy: {valid_accuracy}%")

      else:
         print("\n   Invalid argument provided for set")
         return
   
   # Write execution for csine similarity distance metric
   elif (metric == 'c'):
       
      if (query == 't'):
       # Should take less than 1:10
       test_accuracy = starter.test_knn("Test", "cosim")
       print(f"\n   Test Set Accuracy: {test_accuracy}%")

      elif (query == 'v'):
       # Should take less than 1:10
       valid_accuracy = starter.test_knn("Validation", "cosim")
       print(f"\n   Validation Set Accuracy: {valid_accuracy}%")

      else:
         print("\n   Invalid input provided for set")
         return
   
   # Program should not reach here
   else: 
      print("\n   Invalid input provided for metric")
      return
   
   return

def test_kmeans():
    
   # Print directions
   print("Checking Accuracy & F1-Score of K-Means Function")
   print("**********************************************************")
   print("\n   For euclidean, please input e")
   print("   For cosine similarity, please input c\n")

   metric_input = input("      Please enter a metric to use: ")

   if (metric_input == 'e'):
       metric = 'euclidean'
   elif (metric_input == 'c'):
       metric = 'cosim'
   else:
       print("\n   Invalid argument provided for metric")
       return

   # Print directions
   print("\n   To check validation set accuracy, input v")
   print("   To check test set accuracy, input t\n")

   query_input = input("      Please enter a set to check: ")

   if (query_input == 'v'):
       query = starter.preprocess("mnist_valid.csv")
   elif (query_input == 't'):
       query = starter.preprocess("mnist_test.csv")
   else:
       print("\n   Invalid argument provided for set")
       return
   
   train = starter.preprocess("mnist_train.csv")

   results = starter.kmeans(train, query, metric)
   
  # Initialize the confusion matrix
   matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]
    
   correct_count = 0
   for i in range(len(query)):

        # Get the actual label and prediction
        real_label = int(query[i][0])
        pred_label = int(results[i])

        if real_label == pred_label:
            correct_count += 1

        # Put in the matrix
        matrix[real_label][pred_label] = matrix[real_label][pred_label] + 1

   acc = correct_count / len(query) * 100
   print("\n   Accuracy: " + str(acc))

   f_measure = starter.f_measure(matrix)
   print("   F1-Score: " + str(f_measure))

main()
## Implement a Machine Learning Pipeline

What You'll Learn
- Data Preprocessing: Preparing your data for machine learning
- Model Training: Using a simple algorithm to train a model
- Evaluation: Measuring how well your model performs
- Prediction: Using the model to make predictions on new data

We'll use the following Python libraries:
- `pandas` for data manipulation
- `scikit-learn` for machine learning algorithms and tools
- `matplotlib` for visualization

# Step 1: Setting Up Your Environment
Before we start, ensure you have the necessary libraries installed. You can install them using pip:

`pip install pandas scikit-learn matplotlib`

# Step 2: Loading and Exploring the Data
Every machine learning project starts with data. We'll use the famous **Iris dataset**, which is included in `scikit-learn`.
This dataset contains measurements of 150 iris flowers from three different species. 

Cool, huh? :grin:

![Alt text](https://raw.githubusercontent.com/rb-thompson/machine-learning-basics/refs/heads/main/project_files/print-head.png "printed data head")

## Explanation:

- `iris.data` contains the features (sepal length, sepal width, petal length, petal width).
- `iris.target` contains the labels (species of iris).
- We use `pandas` to organize the data into a tabular format.

# Step 3: Data Processing
Before training a model, we need to preprocess the data. This involves:

1. Splitting the data into features (`X`) and labels (`Y`).
2. Splitting the data into training and testing sets.

![Alt text](https://raw.githubusercontent.com/rb-thompson/machine-learning-basics/refs/heads/main/project_files/test-train-split.jpg "test-train-split method graphic")

## Explanation:

- `train_test_split` randomly splits the data into training and testing sets.
- `random_state=42` ensures repoducibility (you'll get the same split every time).

# Step 4: Training a Machine Learning Model
Now, let's train a simple machine learning model. We'll use the **k-Nearest Neighbors (k-NN) algorithm, 
which is easy to understand and implement.

## Explanation:

- `KNeighborsClassifier` is a simple algorithm that classifies data points based on the majority class among their `k` nearest neighbors.
- `fit` trains the model using the training data.

# Step 5: Evaluating a Model
After training, we need to evaluate the model's performance on the test data. We'll use **accuracy** as our metric.

![Alt text](https://raw.githubusercontent.com/rb-thompson/machine-learning-basics/refs/heads/main/project_files/accuracy-score.png "calculated model accuracy score graphic")

## Explanation:

- `predict` generates predictions for the test data.
- `accuracy_score` compares the predictions to the true labels and calculates the proportion of correct predictions. 


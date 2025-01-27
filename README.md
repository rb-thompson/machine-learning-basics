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

```
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert it to a pandas DataFrame for easier manipulation
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows of the dataset
print(data.head())
```

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

```
from sklearn.model_selection import train_test_split

# Split the data into features (X) and labels (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
```

![Alt text](https://raw.githubusercontent.com/rb-thompson/machine-learning-basics/refs/heads/main/project_files/train-test-split.jpg "test-train-split method graphic")

## Explanation:

- `train_test_split` randomly splits the data into training and testing sets.
- `random_state=42` ensures repoducibility (you'll get the same split every time).


# Step 4: Training a Machine Learning Model
Now, let's train a simple machine learning model. We'll use the **k-Nearest Neighbors** (k-NN) algorithm, 
which is easy to understand and implement.

```
from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)
```

## Explanation:

- `KNeighborsClassifier` is a simple algorithm that classifies data points based on the majority class among their `k` nearest neighbors.
- `fit` trains the model using the training data.

# Step 5: Evaluating a Model
After training, we need to evaluate the model's performance on the test data. We'll use **accuracy** as our metric.

```
from sklearn.metrics import accuracy_score

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

![Alt text](https://raw.githubusercontent.com/rb-thompson/machine-learning-basics/refs/heads/main/project_files/accuracy-score.png "calculated model accuracy score graphic")

## Explanation:

- `predict` generates predictions for the test data.
- `accuracy_score` compares the predictions to the true labels and calculates the proportion of correct predictions. 


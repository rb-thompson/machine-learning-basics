import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# install and import libraries
# `pip install pandas scikit-learn matplotlib`

# Load the Iris dataset
iris = load_iris()

# Convert it to a pandas DataFrame for easier manipulation
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows of the dataset
print(data.head())

# TIP: Get Familiar With The Data
# EXAMPLE:
# ------------------------------------------------------------------------------------
# Upon inspection, this dataset shows various metrics using flower parts.
#
#   SEPAL LENGTH & WIDTH - The green leaf-like structures that encase the flower bud.
#   PETAL LENGTH & WIDTH - The colored structures of the flower.
#   SPECIES - A category of living things.
# ------------------------------------------------------------------------------------

# Split the data into features (X) and labels (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets (80% train, 20% test is common)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Initialize the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
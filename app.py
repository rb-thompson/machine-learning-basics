import pandas as pd
from sklearn.datasets import load_iris

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


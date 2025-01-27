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

![Alt text](https:github.com/rb-thompson/machine-learning-basics/blob/main/project_files/print-head.png "printed data head")

## Explanation:

- `iris.data` contains the features (sepal length, sepal width, petal length, petal width).
- `iris.target` contains the labels (species of iris).
- We use `pandas` to organize the data into a tabular format.
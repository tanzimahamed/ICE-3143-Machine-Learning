# Decision Tree Classifier – Play Tennis

**ICE 3144 – Machine Learning Lab | Lab 03**

This project demonstrates a **Decision Tree Classification** model using the Play Tennis dataset.

## Lab_03

The notebook performs the following steps:

1. Mounts Google Drive in Google Colab
2. Imports the required Python libraries
3. Loads the Play Tennis dataset
4. Explores the dataset using shape, head, tail, and column names
5. Removes the `Day` column
6. Separates features (`X`) and target (`y`)
7. Converts categorical features into numerical values using **One-Hot Encoding**
8. Splits the data into training and testing sets
9. Trains a **Decision Tree Classifier**
10. Visualizes the decision tree
11. Evaluates the model using accuracy

##  Dataset

The dataset contains 14 records showing whether tennis can be played based on different weather conditions.

### Features

- `Outlook` (Sunny / Overcast / Rain)
- `Temperature` (Hot / Mild / Cool)
- `Humidity` (High / Normal)
- `Wind` (Weak / Strong)

### Target

- `Play Tennis` (Yes / No)

The `Day` column is removed before training because it is not used as a prediction feature.

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

##  Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
```

##  Data Preprocessing

### 1. Remove the Day Column

```python
df.drop(columns=['Day'], inplace=True)
```

### 2. Separate Features and Target

```python
X = df.drop('Play Tennis', axis=1)
y = df['Play Tennis']
```

### 3. One-Hot Encoding

Since the dataset contains categorical values, they are converted into numerical values:

```python
X_encoded = pd.get_dummies(X)
```

##  Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)
```

- Training data: 80%
- Testing data: 20%
- `random_state`: 42

##  Decision Tree Model

A `DecisionTreeClassifier` is created and trained using the training data.

```python
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
```

##  Decision Tree Visualization

The trained decision tree is visualized using Scikit-learn and Matplotlib:

```python
plt.figure(figsize=(12, 8))

tree.plot_tree(
    clf,
    feature_names=X_encoded.columns,
    class_names=clf.classes_,
    filled=True
)

plt.show()
```

##  Model Evaluation

The model is evaluated using classification accuracy:

```python
accuracy = clf.score(X_test, y_test)

print("Accuracy:", accuracy)
```

**Result:** Accuracy = `1.0` (100%)

> **Note:** The dataset is very small (14 samples, so only 3 test samples), so the 100% accuracy does not mean the model is strong. This project is mainly for learning purposes.


##  How to Run

### Google Colab

1. Upload `Decision_Tree.ipynb` to Google Colab.
2. Upload `play_tennis_dataset.csv` to the specified Google Drive folder.
3. Run the notebook cells from top to bottom.
4. Check the decision tree visualization and accuracy output.

### Local Jupyter Notebook

If running locally, update the dataset path in the `pd.read_csv()` cell:

```python
df = pd.read_csv("play_tennis_dataset.csv")
```

##  Learning Objectives

This project helps understand:

- Decision Tree Classification
- Categorical data preprocessing
- One-Hot Encoding
- Train-Test Split
- Model Training
- Decision Tree Visualization
- Classification Accuracy

##  Author

**Tanzim Ahamed**

Information and Communication Engineering (ICE)  
Daffodil International University

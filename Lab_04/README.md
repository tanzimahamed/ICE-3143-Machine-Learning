# Candidate-Elimination Algorithm

**ICE 3144 – Machine Learning Lab | Lab 04**

This experiment demonstrates the **Candidate-Elimination Algorithm**, a concept learning algorithm used to find the set of hypotheses consistent with the training examples.

## Lab_04

The notebook performs the following steps:

1. Imports the required Python libraries
2. Loads the training dataset
3. Explores the dataset using shape, head, tail, and column names
4. Separates the attributes and target concept
5. Initializes the **Specific Boundary (S)** and **General Boundary (G)**
6. Processes positive and negative training examples
7. Updates the specific boundary for positive examples
8. Generalizes the specific boundary when necessary
9. Specializes the general boundary for negative examples
10. Removes inconsistent hypotheses
11. Displays the final version space boundaries

## Algorithm Overview

The Candidate-Elimination Algorithm learns a concept by maintaining two boundaries:

- **Specific Boundary (S):** Represents the most specific hypotheses consistent with the training data.
- **General Boundary (G):** Represents the most general hypotheses consistent with the training data.

The algorithm updates these boundaries after processing each training example.

### Positive Example

When the target label is positive (`Yes`):

- Generalize the specific boundary `S` only as much as necessary.
- Remove hypotheses from `G` that do not cover the positive example.

### Negative Example

When the target label is negative (`No`):

- Specialize the general boundary `G` only as much as necessary.
- Remove hypotheses from `S` that incorrectly cover the negative example.

## Dataset

The dataset contains training examples with attributes and a target concept. A common example for this algorithm is the **EnjoySport dataset**, which describes weather and other conditions under which a person enjoys a sport.

### Example Features

Depending on the dataset used in the notebook, the features may include:

- `Sky`
- `AirTemp`
- `Humidity`
- `Wind`
- `Water`
- `Forecast`

### Target

- `EnjoySport` (`Yes` / `No`)

**Note:** Update the feature names and target label above if your notebook uses a different dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Google Colab / Jupyter Notebook

## Required Libraries

```python
import numpy as np
import pandas as pd
```

## Algorithm Implementation

### 1. Initialize the Boundaries

The specific boundary starts with the first positive training example, while the general boundary starts with the most general hypothesis.

```python
S = None
G = [['?' for _ in range(num_attributes)]]
```

*Note: The initialization code may differ depending on the implementation used in the notebook.*

### 2. Process Training Examples

The algorithm processes each training example and updates the boundaries according to whether the example is positive or negative.

- For positive examples, generalize `S` when necessary.
- For negative examples, specialize `G` when necessary.
- Remove hypotheses that are inconsistent with the observed examples.

### 3. Display the Final Boundaries

After processing all training examples, the algorithm returns the final specific and general boundaries.

```python
print("Final Specific Boundary (S):", S)
print("Final General Boundary (G):", G)
```

## Expected Output

The notebook displays:

- The loaded dataset
- The initial specific and general boundaries
- Boundary updates for each training example
- The final specific boundary (`S`)
- The final general boundary (`G`)

The exact output depends on the dataset and the order of the training examples.

## How to Run

### Google Colab

1. Upload the Candidate-Elimination notebook to Google Colab.
2. Upload the required dataset if the notebook uses an external CSV file.
3. Update the dataset path if necessary.
4. Run all cells from top to bottom.
5. Observe how the `S` and `G` boundaries change after each example.

### Local Jupyter Notebook

1. Install Python and Jupyter Notebook.
2. Install the required libraries:

```bash
pip install numpy pandas
```

3. Place the notebook and dataset in the appropriate directory.
4. Run the notebook cells in order.

## Learning Objectives

This experiment helps understand:

- Concept Learning
- Candidate-Elimination Algorithm
- Version Space
- Specific and General Boundaries
- Positive and Negative Training Examples
- Generalization and Specialization of Hypotheses
- Hypothesis Consistency

## Author

**Tanzim Ahamed**

Information and Communication Engineering (ICE)  
Daffodil International University

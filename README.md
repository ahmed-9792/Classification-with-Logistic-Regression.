Binary Classification with Logistic Regression

📌 Objective:
Build and evaluate a binary classification model using Logistic Regression to classify breast cancer tumors as Malignant (M) or Benign (B).

🛠 Tools & Libraries:
Python 3

Pandas – data handling

NumPy – numerical operations

Scikit-learn – machine learning

Matplotlib – visualization

📂 Dataset:
The dataset used is the Breast Cancer Wisconsin Dataset (data.csv).

Columns:
Target: diagnosis

M = Malignant (1)

B = Benign (0)

Features: 30 numeric features describing tumor characteristics.

Dropped Columns:

id (identifier)

Unnamed: 32 (empty column)

📋 Steps Performed:

Load the dataset from CSV.

Drop irrelevant columns (id, Unnamed: 32).

Encode target variable (diagnosis: M=1, B=0).

Split data into train (80%) and test (20%) sets with stratification.

Standardize features using StandardScaler.

Train Logistic Regression model with max_iter=500.

Evaluate performance:

Confusion matrix

Classification report (precision, recall, F1-score)

ROC curve and AUC score

Threshold tuning example to adjust classification sensitivity.

Explain sigmoid function in the context of Logistic Regression.

📊 Model Performance:

Default Threshold (0.5):

Accuracy: ~96%

ROC-AUC: ~0.996

Confusion Matrix:

[[71,  1],
 [ 3, 39]]
With Threshold = 0.4:
Used to increase sensitivity for malignant detection.

📈 ROC Curve:

The ROC curve was plotted to visualize the trade-off between True Positive Rate (Recall) and False Positive Rate.

🔢 Sigmoid Function:

Logistic Regression uses:

sigmoid(z) = 1 / (1 + e^(-z))
This maps model outputs to probabilities between 0 and 1, allowing us to classify based on a chosen threshold.

▶ How to Run:
Install dependencies:

pip install pandas numpy scikit-learn matplotlib
Place data.csv in the working directory.

Run the Python script or Jupyter Notebook:

python logistic_regression_task4.py
or

jupyter notebook TASK4.ipynb

📁 Files:

data.csv – Dataset file.

TASK4.ipynb – Jupyter Notebook implementation.

logistic_regression_task4.py – Python script version.

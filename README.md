# Datheon-Internship-Tasks
#Task 1
Multi-Class Benchmark Classification Pipeline --- California Housing
Project Overview
This project implements an end-to-end multi-class classification
pipeline using the supplied California Housing dataset and Scikit-Learn.

The original median_house_value column contains continuous house
prices. To make the problem suitable for multi-class classification, it
is converted into three project-defined categories:

  Category                      House value
  Low                       Below $150,000
Medium       $150,000 to below $300,000
High                  $300,000 and above

The notebook compares Logistic Regression and Decision Tree
classifiers using accuracy, precision, recall, F1-score, classification
reports, and confusion matrices.
Project Files
- California_Housing_Multi_Class_Classification_Pipeline.ipynb ---
  main Jupyter Notebook.
- california_housing.csv --- dataset used by the notebook.
- README.md --- project documentation.
Keep the notebook and CSV file in the same folder.
Objectives
- Load and explore the dataset using Pandas.
- Inspect missing values and descriptive statistics.
- Visualize feature relationships with a Seaborn pairplot.
- Convert continuous house values into Low, Medium, and High classes.
- Split data into training and testing sets using an 80/20 split.
- Impute missing values, standardize numeric features, and encode
  categorical features.
- Train Logistic Regression and Decision Tree classifiers.
- Compare classification metrics.
- Plot confusion matrix heatmaps.
- Visualize model decision regions using PCA.
Technologies and Libraries
- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
Dataset Features
The dataset includes:
- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- median_house_value
- ocean_proximity
median_house_value is used to create the classification target. The
original continuous target column is excluded from model input features
to prevent target leakage.
Methodology
1. Data Loading and Exploration
The CSV file is loaded with Pandas. The notebook displays sample rows,
dataset information, descriptive statistics, missing-value counts, class
counts, and a pairplot based on a reproducible sample of up to 1,000
rows.
2. Target Engineering
The continuous median_house_value target is grouped into three
classes:
- Low: less than $150,000
- Medium: $150,000 to less than $300,000
- High: $300,000 or more
These thresholds are introduced for this classification task; they are
not original labels supplied by the dataset.
3. Train/Test Split
The data is divided into 80% training data and 20% testing data. A
stratified split preserves class proportions. The random state is set to
42 for reproducibility.
4. Preprocessing
- Numeric columns: median imputation followed by StandardScaler.
- Categorical columns: most-frequent imputation followed by
  OneHotEncoder.
- Preprocessing is fitted on training data only and then applied to
  test data.
5. Model Training
Two supervised classification algorithms are trained:
- Logistic Regression
- Decision Tree Classifier
6. Model Evaluation
The notebook reports accuracy, weighted precision, weighted recall, and
weighted F1-score. It also prints per-class precision, recall, F1-score,
and support through classification_report.

Confusion matrix heatmaps are plotted for both models.
7. Decision Region Visualization
PCA projects the processed feature space onto two dimensions for
visualization. The classifiers are trained on the full processed feature
set; PCA is used to display their decision regions, not to replace the
original training features.

#Task2
California Housing Price Prediction --- OLS, Ridge & Lasso
Project Overview
This project predicts California median house values using the
California Housing dataset. It compares three regression approaches:
- Ordinary Least Squares (OLS) using LinearRegression
- Ridge Regression (L2 regularization) with hyperparameter tuning
- Lasso Regression (L1 regularization) with hyperparameter tuning
The notebook also performs feature engineering, missing-value
imputation, categorical encoding, feature scaling, correlation analysis,
and residual error analysis.
Dataset
The project uses the supplied housing.csv dataset (20,640 rows and 10
original columns).

The target variable is:
- median_house_value
The notebook first looks for housing.csv in the same folder. If it is
not present, it attempts to download the public California Housing CSV
from:

https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
Project Files
California_Housing_Regression/
├── California_Housing_Ridge_Lasso_Regression.ipynb
├── housing.csv
└── README.md
Requirements
- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy (optional; used for smooth residual density curves)
Install the required packages:
pip install pandas numpy matplotlib scikit-learn scipy jupyter
How to Run
1. Download and extract the project ZIP file.
2. Keep housing.csv, the notebook, and this README in the same
   folder.
3. Open a terminal or command prompt in that folder.
4. Start Jupyter:
   jupyter notebook
5. Open California_Housing_Ridge_Lasso_Regression.ipynb.
6. Run the notebook cells from top to bottom.
You can also upload the notebook and CSV to Google Colab. If using
Colab, upload housing.csv into the notebook's working directory before
running the cells.
Workflow
1. Load the dataset and inspect its structure.
2. Check missing values and duplicate rows.
3. Calculate a numeric-feature correlation matrix and display a
   heatmap.
4. Engineer household ratios and interaction features:
   - rooms per household
   - bedrooms per room
   - population per household
   - income per room
   - income × rooms per household
   - housing age × income
5. Split data into training and test sets.
6. Impute missing numeric values using the median.
7. Impute missing categorical values using the most frequent category.
8. Standardize numeric features and one-hot encode categorical
   features.
9. Fit OLS and tune Ridge and Lasso alpha values with 5-fold
   GridSearchCV.
10. Compare test-set RMSE, MAE, and R².
11. Plot predicted vs. actual values and inspect residual distributions
    and residuals against predictions.
12. Compare Ridge and Lasso coefficients.
Evaluation Metrics
- RMSE (Root Mean Squared Error): penalizes larger prediction
  errors more strongly. Lower is better.
- MAE (Mean Absolute Error): average absolute prediction error.
  Lower is better.
- R² (Coefficient of Determination): measures the proportion of
  target variance explained by the model. Higher is generally better.
The notebook displays the actual metric values after its cells have been
executed. No performance values are hard-coded in this README.
Notes and Limitations
- Preprocessing is placed inside scikit-learn pipelines, so
  imputation, scaling, and encoding are fitted on training data rather
  than the held-out test set.
- Ridge and Lasso are sensitive to feature scale, so numeric features
  are standardized.
- The dataset's commonly used median_house_value target has an upper
  cap. This can affect predictions and residual patterns near the
  maximum target value.
- Correlation is a diagnostic for linear association; it does not
  establish causation.
- The notebook uses a random 80/20 train-test split with
  random_state=42 for reproducibility.

  # Task3
  # Predictive Customer Churn Classifier
### Customer Churn Prediction using Random Forest & XGBoost

## Project Overview
This project predicts whether a telecom customer is likely to leave the company (churn) using Machine Learning algorithms.

Two classification models are implemented:
- Random Forest Classifier
- XGBoost Classifier

The project uses data preprocessing, One-Hot Encoding, class imbalance handling, cross-validation, and model evaluation techniques to build and compare predictive models.

## Dataset
**Dataset:** Telco Customer Churn

The dataset contains 7,043 customer records and 21 columns, including:
- Customer demographics
- Account information
- Services subscribed
- Contract details
- Monthly and total charges
- Churn status

**Target Variable:** `Churn`
- `Yes` = Customer left the company
- `No` = Customer stayed with the company

## Technologies & Libraries
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost

## Project Workflow

### 1. Data Loading & Exploration
- Load the telecom churn dataset.
- Explore dataset dimensions and columns.
- Analyze churn distribution.
- Identify missing values.

### 2. Data Preprocessing
- Convert `TotalCharges` into numeric format.
- Handle missing values.
- Remove irrelevant `customerID` column.
- Encode the target variable.
- Split data into training and testing sets.

### 3. Feature Engineering
- Separate numerical and categorical features.
- Apply One-Hot Encoding to categorical variables.
- Apply scaling to numerical features.
- Use a preprocessing pipeline.

### 4. Handling Class Imbalance
- Apply SMOTE with the Random Forest pipeline.
- Use class weighting through `scale_pos_weight` with XGBoost.

### 5. Model Training
Train and compare:
- Random Forest Classifier
- XGBoost Classifier

### 6. Cross-Validation
Use Stratified 5-Fold Cross-Validation to evaluate model performance using ROC-AUC.

### 7. Model Evaluation
Evaluate models using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Classification Report

### 8. Visualizations
Generate:
- Churn distribution plot
- ROC Curve
- Confusion Matrix
- Top 10 Feature Importance plot

## Project Structure

```text
Predictive-Customer-Churn/
│
├── Predictive_Customer_Churn_Classifier.ipynb
├── Telco-Customer-Churn.csv
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Predictive-Customer-Churn
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

## How to Run

1. Download or clone this repository.
2. Keep the notebook and CSV dataset in the same folder.
3. Open Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Predictive_Customer_Churn_Classifier.ipynb`.
5. Run all cells sequentially.

## Expected Results

The project compares Random Forest and XGBoost using cross-validation and test-set evaluation.

The notebook generates:
- Model performance metrics
- ROC curves
- Confusion matrices
- Top 10 important features

**Note:** Actual model performance depends on the dataset and execution environment. No specific accuracy or ROC-AUC score is guaranteed.

## Key Learning Outcomes

- Data preprocessing and feature engineering
- One-Hot Encoding
- Handling imbalanced datasets
- SMOTE and class weighting
- Random Forest and XGBoost
- Stratified Cross-Validation
- ROC-AUC evaluation
- Feature importance analysis


# task 4
# NLP Sentiment Classification on Customer Reviews (TF-IDF)

## Project Overview
This project implements a Natural Language Processing (NLP) pipeline to classify customer reviews into positive and negative sentiments using TF-IDF vectorization and machine learning algorithms.

## Objectives
- Preprocess and clean customer review text.
- Convert text into numerical features using TF-IDF.
- Train sentiment classification models.
- Compare Multinomial Naive Bayes and Logistic Regression.
- Predict sentiment for custom user input.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK / Regex
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Machine Learning Algorithms
1. Multinomial Naive Bayes
2. Logistic Regression

## Project Workflow
1. Load customer review dataset.
2. Clean text by removing HTML tags, punctuation, numbers, and stopwords.
3. Convert text into TF-IDF feature vectors.
4. Split data into training and testing sets.
5. Train both classification models.
6. Evaluate models using accuracy, classification reports, and confusion matrices.
7. Predict sentiment using custom review text.

## Dataset
The project includes the 10 customer reviews provided in the assignment.

Dataset source:
https://www.datheon.in/datasets/sentiment_reviews.csv

The sample contains positive, negative, and neutral reviews. Neutral reviews are excluded from binary model training.

## Project Structure
```text
NLP-Sentiment-Classification/
│
├── NLP_Sentiment_Classification_TFIDF.ipynb
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## How to Run
1. Download the project notebook.
2. Open it using Jupyter Notebook or Google Colab.
3. Install the required libraries.
4. Run all notebook cells.
5. Enter custom customer reviews in the interactive prediction section.

## Prediction Example

Input:
"I absolutely love this product. It works perfectly!"

Expected output:
Positive

Input:
"The product stopped working and customer service was terrible."

Expected output:
Negative

## Evaluation Metrics
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Conclusion
This project demonstrates how NLP and machine learning can be used to analyze customer feedback and automatically classify review sentiment. TF-IDF extracts important textual features, while Naive Bayes and Logistic Regression perform sentiment classification.



  

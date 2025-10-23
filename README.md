# Multiple-Linear-Regression-Model-for-Sales-Prediction

## 1. Overview and Goal
This project implements a Multiple Linear Regression (MLR) model in Python to predict sales revenue based on advertising expenditures across three media channels: TV, Radio, and Newspaper. The goal is to understand the correlation between spending and sales, quantify the influence of each channel, and build a highly accurate predictive tool.

## 2. Dataset
* Source: ***advertising.csv*** (A standard public dataset used for regression analysis).
* Features ($\mathbf{X}$): ***TV, Radio,*** and ***Newspaper*** (Budgets, typically in thousands of currency units).
* Target ($\mathbf{Y}$): ***Sales*** (Revenue, typically in thousands of currency units).

## 3. Project Steps and Code Breakdown
The analysis follows standard machine learning and statistical modeling procedures:
### Step 1: Data Inspection and Visualization (EDA)
* Action: Loads the data, checks for missing values (df.info()), and displays the first few rows.
* Visualization: Uses sns.pairplot to visualize the linear relationship between each feature and the target (Sales) individually, which helps confirm the initial assumption of linearity.
### Step 2: Data Preparation
* Action: Separates the data into the feature matrix $\mathbf{X}$ (the three budget columns) and the target vector $\mathbf{Y}$ (the Sales column).
* Splitting: Uses train_test_split with a test_size=0.2 (20% for testing) and random_state=42 (for reproducibility).
### Step 3: Model Training
* Action: Initializes the LinearRegression() model and trains it using the split training data (X_train, Y_train).
### Step 4: Model Evaluation and Interpretation
* Prediction: Generates Y_pred values by applying the trained model to the unseen test data (X_test).
* Metrics Calculation: Calculates the two core metrics:
  * Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
  * R-squared ($\mathbf{R^2}$): Measures the proportion of the variance in the target variable that is predictable from the features.
* Coefficient Analysis: Extracts the Intercept ($\beta_0$) and Coefficients ($\beta_1$, $\beta_2$, $\beta_3$) to form the final predictive equation and interpret the impact of each advertising channel.

| Metric | Result | Interpretation |
| --- | --- | --- |
| R-squared ($\mathbf{R^2}$) | $\approx 0.9059$ | 90.59% of the Sales variance is explained by the advertising budgets. |
| MSE | $\approx 2.91$ | Low error value, indicating accurate predictions. |

### Step 5: Visualizations
* Correlation Heatmap: A visual summary showing the strength of the linear relationship between every pair of variables.
* Actual vs. Predicted Plot: A scatter plot comparing the Y_test (Actual Sales) against the Y_pred (Predicted Sales). The close proximity of the points to the diagonal "Ideal Fit" line confirms the model's high accuracy.

# 4. Running the Project
* Dependencies: Ensure you have Python and the following libraries installed:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn
```
* Data File: Place the advertising.csv file in the same directory as your Python script.
* Execution: Run the Python script or notebook. The output will display the model's parameters, coefficients, and performance metrics.

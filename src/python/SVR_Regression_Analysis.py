import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from kernel_methods import rbf_kernel, polynomial_kernel, svm_with_custom_kernel

# Load the preprocessed data
file_path = 'Preprocessed_Data (1).xlsx'
preprocessed_data = pd.read_excel(file_path)

# Separate features and target variable
features = preprocessed_data.drop(['score', 'ProbeID', 'GeneSymbol'], axis=1)
target = preprocessed_data['score']

# Feature Selection using Mutual Information
mutual_info = mutual_info_regression(features, target)
mutual_info_dict = dict(zip(features.columns, mutual_info))

# Correlation Matrix
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=False)
plt.show()

# Recursive Feature Elimination (RFE)
selector = RFE(SVR(kernel="linear"), n_features_to_select=5, step=1)
selector = selector.fit(features, target)
features_selected_by_rfe = features.columns[selector.support_]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Model Training with SVR
svm_model = SVR(kernel="linear")
svm_model.fit(X_train, y_train)

# Model Evaluation
y_pred = svm_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred)
rmse_svr = np.sqrt(mse_svr)

# Baseline Model for comparison
dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(X_train, y_train)
y_dummy = dummy_regressor.predict(X_test)
mse_dummy = mean_squared_error(y_test, y_dummy)
rmse_dummy = np.sqrt(mse_dummy)

# Cross-Validation of SVR model
svr_rmse_cv = np.sqrt(-cross_val_score(svm_model, features, target, cv=5, scoring='neg_mean_squared_error'))

# Dictionary of models for cross-validation
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net Regression': ElasticNet(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor()
}

# Cross-validation of different models
cv_results = {model_name: np.sqrt(-cross_val_score(model, features, target, cv=5, scoring='neg_mean_squared_error'))
              for model_name, model in regression_models.items()}

# Summary of cross-validation results
cv_results_summary = {model_name: {'Mean RMSE': np.mean(scores), 'Std RMSE': np.std(scores)}
                      for model_name, scores in cv_results.items()}

# Printing the results
print("Mutual Information:", mutual_info_dict)
print("Features selected by RFE:", features_selected_by_rfe)
print(f"SVR Mean Squared Error: {mse_svr}")
print(f"SVR Root Mean Squared Error: {rmse_svr}")
print(f"Baseline Model Mean Squared Error: {mse_dummy}")
print(f"Baseline Model Root Mean Squared Error: {rmse_dummy}")
print("Cross-validated RMSE scores for SVR model:", svr_rmse_cv)
print(cv_results_summary)

# Note: The heatmap is generated but not displayed in this script output.

# Use custom RBF kernel
rbf_model = svm_with_custom_kernel(X_train, y_train, rbf_kernel, gamma=0.1)
rbf_predictions = rbf_model.predict(X_test)

# Use custom polynomial kernel
poly_model = svm_with_custom_kernel(X_train, y_train, polynomial_kernel, degree=3, coef0=1)
poly_predictions = poly_model.predict(X_test)

# Evaluate models
print("RBF Kernel MSE:", mean_squared_error(y_test, rbf_predictions))
print("RBF Kernel R2:", r2_score(y_test, rbf_predictions))
print("Polynomial Kernel MSE:", mean_squared_error(y_test, poly_predictions))
print("Polynomial Kernel R2:", r2_score(y_test, poly_predictions))

from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # SVR is used for regression tasks
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the preprocessed data
# Please make sure the file path is correct and accessible from your Python environment.
preprocessed_data = pd.read_excel("path_to_Preprocessed_Data.xlsx", index_col=0)

# Separate features and target variable
features = preprocessed_data.drop(['score', 'ProbeID', 'GeneSymbol'], axis=1)
target = preprocessed_data['score']

# 1. Feature Selection
## 1.1 Mutual Information
mutual_info = mutual_info_regression(features, target)
mutual_info_dict = dict(zip(features.columns, mutual_info))
print("Mutual Information:", mutual_info_dict)

## 1.2 Correlation Matrix
correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
selector = RFE(SVR(kernel="linear"), n_features_to_select=5, step=1)
selector = selector.fit(features, target)
print("Features selected by RFE:", features.columns[selector.support_])

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 3. Model Training with Support Vector Machine for Regression
svm_model = SVR(kernel="linear")  # Specify the kernel if needed, default is 'rbf'
svm_model.fit(X_train, y_train)

# Model Evaluation
y_pred = svm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

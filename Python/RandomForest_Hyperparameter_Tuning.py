import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Assuming 'preprocessed_data' is already loaded and available
# Replace 'path_to_your_data.xlsx' with the actual file path
# preprocessed_data = pd.read_excel('path_to_your_data.xlsx')

# Separate features and target variable
# Make sure the column names match those in your dataset
features = preprocessed_data.drop(['score', 'ProbeID', 'GeneSymbol'], axis=1)
target = preprocessed_data['score']

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],  # The number of trees in the forest.
    'max_depth': [None, 10, 20, 30],  # The maximum depth of the trees.
    'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node.
    'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node.
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees.
}

# Create a base model
rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(features, target)

# Print the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", np.sqrt(-grid_search.best_score_))

# Save the best model
best_model = grid_search.best_estimator_

# Save the model to a file (optional)
# from joblib import dump
# dump(best_model, 'best_random_forest_model.joblib')

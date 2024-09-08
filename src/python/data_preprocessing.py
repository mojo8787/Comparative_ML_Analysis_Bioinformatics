import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def load_data(features_path, target_path):
    """Load features and target data from Excel files."""
    features = pd.read_excel(features_path)
    target = pd.read_excel(target_path)
    return features, target['Target']

def preprocess_features(features):
    """Preprocess features: handle missing values and scale."""
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features.columns)
    
    return features_scaled

def select_features(features, target, k=10):
    """Select top k features based on f_regression."""
    selector = SelectKBest(score_func=f_regression, k=k)
    selected_features = selector.fit_transform(features, target)
    selected_feature_names = features.columns[selector.get_support()].tolist()
    
    return pd.DataFrame(selected_features, columns=selected_feature_names)

def main():
    # Load data
    features, target = load_data('../../Data/raw/features.xlsx', '../../Data/raw/target.xlsx')
    
    # Preprocess features
    features_preprocessed = preprocess_features(features)
    
    # Select top features
    features_selected = select_features(features_preprocessed, target)
    
    # Save preprocessed and selected features
    features_preprocessed.to_csv('../../Data/processed/features_preprocessed.csv', index=False)
    features_selected.to_csv('../../Data/processed/features_selected.csv', index=False)
    
    print(f"Preprocessing complete. Preprocessed features shape: {features_preprocessed.shape}")
    print(f"Feature selection complete. Selected features shape: {features_selected.shape}")

if __name__ == "__main__":
    main()
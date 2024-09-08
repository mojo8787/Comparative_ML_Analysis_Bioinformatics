library(tidyverse) # For data manipulation and visualization
library(caret) # For machine learning
library(e1071) # For support vector machines
library(rpart) # For decision trees
library(readxl) # For reading Excel files
library(randomForest) # For the random forest algorithm used in RFE
library(readr) # For reading CSV files
library(dplyr) # For data manipulation
library(ggplot2) # For data visualization

# Set the file path and load the preprocessed data
file_path <- "/Users/Saham/desktop/NEW_ML/Preprocessed_Data (1).xlsx"
preprocessed_data <- read_excel(file_path)

# Separate features and target variable
features <- preprocessed_data %>% select(-c("score", "ProbeID", "GeneSymbol"))
target <- preprocessed_data$score

# Ensure all features are numeric for correlation analysis
numeric_features <- select_if(features, is.numeric)
correlation_matrix <- cor(numeric_features)
ggplot(as.data.frame(as.table(correlation_matrix)), aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2()

# Define control function using random forest with repeated cross-validations for RFE
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10, repeats = 3)

# Run RFE
rfe_results <- rfe(numeric_features, target, sizes = c(1:5), rfeControl = control)

# Print the results of RFE
print(rfe_results)

# Plot the RFE results
plot(rfe_results, type = c("o", "g"))

# Train-Test Split
set.seed(42) # For reproducibility
training_indices <- createDataPartition(target, p = .70, list = FALSE)
X_train <- numeric_features[training_indices, ]
X_test <- numeric_features[-training_indices, ]
y_train <- target[training_indices]
y_test <- target[-training_indices]

# Model Training with SVR
svm_model <- svm(y_train ~ ., data = X_train, type = "eps-regression", kernel = "linear")

# Model Evaluation
y_pred <- predict(svm_model, newdata = X_test)
mse_svr <- mean((y_test - y_pred)^2)
rmse_svr <- sqrt(mse_svr)

# Baseline Model for comparison
dummy_model <- train(y_train ~ 1, data = X_train, method = "mean")
y_dummy <- predict(dummy_model, newdata = X_test)
mse_dummy <- mean((y_test - y_dummy)^2)
rmse_dummy <- sqrt(mse_dummy)

# Cross-Validation of SVR model
svm_cv <- train(y_train ~ ., data = X_train, method = "svmLinear", trControl = trainControl(method = "cv", number = 5))

# Print the results including the RFE and model evaluation
print(list(Features_selected_by_RFE = predictors(rfe_results)))
print(list(SVR_Mean_Squared_Error = mse_svr))
print(list(SVR_Root_Mean_Squared_Error = rmse_svr))
print(list(Baseline_Model_Mean_Squared_Error = mse_dummy))
print(list(Baseline_Model_Root_Mean_Squared_Error = rmse_dummy))
print(list(Cross_validated_RMSE_scores_for_SVR_model = svm_cv$results$RMSE))

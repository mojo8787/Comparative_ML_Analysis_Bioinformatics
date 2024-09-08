library(tidyverse)        # For data manipulation and visualization
library(caret)            # For machine learning
library(e1071)            # For support vector machines
library(readxl)           # For reading Excel files

# Load the features and target datasets
features_path <- '/Users/Saham/desktop/NEW_ML/features.xlsx'
target_path <- '/Users/Saham/desktop/NEW_ML/target.xlsx'
features <- read_excel(features_path)
target <- read_excel(target_path)

# Combine features and target for splitting
data <- bind_cols(features, target)

# Cross-Validation Setup
set.seed(42) # For reproducibility
train_control <- trainControl(method = "cv", number = 10, search = "grid")

# Hyperparameter Grid
svm_grid <- expand.grid(
  C = seq(0.1, 1, by = 0.1), 
  sigma = seq(0.001, 0.01, by = 0.001)
)

# Model Training with Cross-Validation and Hyperparameter Tuning
svm_model <- train(
  score ~ ., 
  data = data,
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = svm_grid,
  preProcess = c("center", "scale"), # Preprocessing steps
  metric = "RMSE"
)

# Model Evaluation on the final selected model
y_pred <- predict(svm_model, newdata = data)
mse_svr <- mean((data$score - y_pred)^2)
rmse_svr <- sqrt(mse_svr)

# Printing the results
print(list(SVR_Mean_Squared_Error = mse_svr))
print(list(SVR_Root_Mean_Squared_Error = rmse_svr))

# Save the model if needed
saveRDS(svm_model, file = "svm_model.rds")

# Optionally, print the model summary
print(summary(svm_model))

# Save the actual and predicted values to a dataframe
results_df <- tibble(Actual = data$score, Predicted = y_pred)

# Save the results to a CSV file
write_csv(results_df, "/Users/Saham/desktop/NEW_ML/predictions3.csv")

# Plot actual vs. predicted values and learning curves
ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Actual vs Predicted", x = "Actual Score", y = "Predicted Score")

# Learning Curve Plot
learning_curve_df <- svm_model$results
ggplot(learning_curve_df, aes(x = C, y = RMSE)) +
  geom_line() +
  labs(title = "Learning Curve", x = "Cost (C)", y = "RMSE")

# View the first few results
head(results_df)

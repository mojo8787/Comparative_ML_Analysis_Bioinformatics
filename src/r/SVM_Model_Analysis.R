library(tidyverse) # For data manipulation and visualization
library(caret) # For machine learning
library(e1071) # For support vector machines
library(readxl) # For reading Excel files
library(readr) # For reading CSV files
library(ggplot2) # For data visualization

# Read preprocessed data
features_preprocessed <- read_csv("../../Data/processed/features_preprocessed.csv")
features_selected <- read_csv("../../Data/processed/features_selected.csv")
target <- read_excel("../../Data/raw/target.xlsx")

# Combine features and target
data_preprocessed <- cbind(features_preprocessed, Target = target$Target)
data_selected <- cbind(features_selected, Target = target$Target)

# Function to evaluate model performance
evaluate_model <- function(model, test_data) {
  predictions <- predict(model, test_data)
  mse <- mean((test_data$Target - predictions)^2)
  rmse <- sqrt(mse)
  r2 <- 1 - (sum((test_data$Target - predictions)^2) / sum((test_data$Target - mean(test_data$Target))^2))
  return(list(RMSE = rmse, R2 = r2))
}

# Split data into training and testing sets
set.seed(42)
train_index <- createDataPartition(data_selected$Target, p = 0.8, list = FALSE)
train_data <- data_selected[train_index, ]
test_data <- data_selected[-train_index, ]

# Train SVM model
svm_model <- svm(Target ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)

# Evaluate SVM model
svm_performance <- evaluate_model(svm_model, test_data)

# Cross-validation
ctrl <- trainControl(method = "cv", number = 5)
svm_cv <- train(Target ~ .,
  data = train_data, method = "svmRadial",
  trControl = ctrl, tuneLength = 5
)

# Print results
cat("SVM Model Performance:\n")
print(svm_performance)
cat("\nCross-validation Results:\n")
print(svm_cv$results)

# Actual vs Predicted Plot
predictions <- predict(svm_model, test_data)
plot_data <- data.frame(Actual = test_data$Target, Predicted = predictions)

ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "SVM - Actual vs Predicted", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

ggsave("../../images/svm_actual_vs_predicted.png", width = 8, height = 6)

# Feature Importance (using linear kernel SVM)
svm_linear <- svm(Target ~ ., data = train_data, kernel = "linear", cost = 1)
feature_importance <- abs(t(svm_linear$coefs) %*% svm_linear$SV)
feature_importance_df <- data.frame(
  Feature = colnames(train_data)[-ncol(train_data)],
  Importance = as.vector(feature_importance)
)
feature_importance_df <- feature_importance_df[order(feature_importance_df$Importance, decreasing = TRUE), ]

ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "SVM - Feature Importance", x = "Features", y = "Importance") +
  theme_minimal()

ggsave("../../images/svm_feature_importance.png", width = 10, height = 8)

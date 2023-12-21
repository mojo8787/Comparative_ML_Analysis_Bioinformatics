library(ggplot2)
library(readr)
library(ggpubr)

# Load the predictions data
predictions <- read_csv("/Users/Saham/Desktop/Applications_scripts\ /Toxicity-of-TiO2/predictions3.csv")

# Scatter Plot: Actual vs Predicted Values
ggplot(predictions, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Values", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# Density Plot: Distribution of Actual and Predicted Values
ggplot(predictions) +
  geom_density(aes(x = Actual), fill = "blue", alpha = 0.5) +
  geom_density(aes(x = Predicted), fill = "green", alpha = 0.5) +
  labs(title = "Density of Actual and Predicted Values", x = "Values", y = "Density") +
  theme_minimal()

# Cumulative Distribution Plot
ggplot(predictions, aes(x = Actual, y = ..ecdf..)) +
  stat_ecdf(geom = "step", color = "blue") +
  stat_ecdf(aes(x = Predicted), geom = "step", color = "green") +
  labs(title = "Cumulative Distribution of Actual and Predicted Values", x = "Values", y = "Cumulative Distribution") +
  theme_minimal()



# Q-Q Plot with ggplot2
predictions$samples <- seq_along(predictions$Actual)
qq_data <- data.frame(sample_quantiles = quantile(predictions$Predicted, probs = ppoints(predictions$samples)),
                      theoretical_quantiles = quantile(predictions$Actual, probs = ppoints(predictions$samples)))

ggplot(qq_data, aes(x = theoretical_quantiles, y = sample_quantiles)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Q-Q Plot of Actual vs Predicted Values",
       x = "Theoretical Quantiles of Actual",
       y = "Sample Quantiles of Predicted") +
  theme_minimal()


# Violin Plot
ggplot(predictions) +
  geom_violin(aes(x = factor(0), y = Actual), fill = "blue", alpha = 0.5) +
  geom_violin(aes(x = factor(1), y = Predicted), fill = "green", alpha = 0.5) +
  labs(title = "Violin Plot of Actual and Predicted Values", x = "", y = "Values") +
  theme_minimal()

# Calculate residuals
predictions$residuals <- predictions$Actual - predictions$Predicted

# Residuals Plot
ggplot(predictions, aes(x = Predicted, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals Plot", x = "Predicted Values", y = "Residuals") +
  theme_minimal()

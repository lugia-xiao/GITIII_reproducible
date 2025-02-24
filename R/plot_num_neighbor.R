# Load necessary library
library(ggplot2)

# Create the data frame
df <- data.frame(
  model = c(10, 100, 25, 5, 50, 75),
  val_loss_interaction = c(0.076987507, 0.076534524, 0.07663682, 0.077758651, 0.076562859, 0.076547829)
)

# Compute variance_explained
df$variance_explained <- 0.0900561940734801 - df$val_loss_interaction

# Create the line plot
ggplot(df, aes(x = model, y = variance_explained)) +
  geom_line() +
  geom_point() +
  labs(
    x = "Number of neighbors",
    y = "Variance Explained",
  ) +
  theme_minimal()

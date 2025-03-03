library(ggplot2)

# Define the data
data <- data.frame(
  Method = c("baseline-GITIII", "2_layers", "3_layers", "4_layers"),
  Value = c(0.013606194, 0.013746194, 0.013976194, 0.013986194)
)

# Rank methods from high to low
#data$Method <- factor(data$Method, levels = data$Method[order(-data$Value)])

# Create the plot
ggplot(data, aes(x = Method, y = Value, fill = Method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 5)), vjust = -0.5) +  # Add values on bars
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)  # Rotate x-axis text
  ) +
  xlab("Method") + 
  ylab("Variance Explained") +
  ggtitle("Layers")

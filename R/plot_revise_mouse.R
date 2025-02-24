library(ggplot2)

# Define the data
data <- data.frame(
  Method = c("GITIII", "NCEM-GCN", "HoloNet", "COMMOT", "GAT", "GT"),
  Value = c(0.032, 0.02658, 0.006266307, 0.002845295, 0.026285161, 0.03)
)

# Rank methods from high to low
data$Method <- factor(data$Method, levels = data$Method[order(-data$Value)])

# Create the plot
ggplot(data, aes(x = Method, y = Value, fill = Method)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 5)), vjust = -0.5) +  # Add values on bars
  theme_minimal() +
  theme(legend.position = "none") +
  xlab("Method") + 
  ylab("Variance Explained") +
  ggtitle("MERFISH mouse PMC")

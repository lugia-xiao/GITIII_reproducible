library(ggplot2)

# Define the data
data <- data.frame(
  Method = c("GITIII", "NCEM-GCN", "HoloNet", "COMMOT", "GT", "GAT"),
  Value = c(0.0156, 0.010803346, 0.00948, 0.009453542, 0.011982026, 0.010425591)
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
  ggtitle("Cosmx human NSCLC")

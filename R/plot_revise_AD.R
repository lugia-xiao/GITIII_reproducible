library(ggplot2)

# Define the data
# Define the data
data <- data.frame(
  Method = c("GITIII", "NCEM-GCN", "HoloNet", "GAT", "GT"),
  Value = c(0.04119907, 0.030205363, 0.028988718, 0.031028945, 0.038221493)
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
  ggtitle("MERFISH human AD MTG")

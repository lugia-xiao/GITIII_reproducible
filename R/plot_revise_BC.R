library(ggplot2)

# Define the data
data <- data.frame(
  Method = c("GITIII", "NCEM-GCN", "HoloNet", "COMMOT", "GT", "GAT"),
  Value = c(0.0307, 0.010221047, 0.00258, 0.00145693, 0.014722472, 0.009176866)
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
  ggtitle("Xenium human breast cancer")

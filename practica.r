# Crear un DataFrame de ejemplo
data <- data.frame(
  Name = c("Alice", "Bob", "Charlie", "David", "Eva", "Frank"),
  Group = c("A", "A", "B", "B", "A", "B"),
  Score = c(85, 90, 78, 88, 92, 80)
)

# Mostrar el DataFrame original
print(data)

# Calcular el puntaje promedio por grupo usando ave
data$AverageScore <- ave(data$Score, data$Group, FUN = mean)

# Mostrar el DataFrame con el puntaje promedio
print(data)


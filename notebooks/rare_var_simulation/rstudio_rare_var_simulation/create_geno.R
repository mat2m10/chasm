# Load necessary functions
source("association.R")
dyn.load("kpop.so")

# Function to simulate genotypes and save the data as CSV
simulate_and_save_csv <- function(G, L, c, k, M) {
  # Simulate genotypes
  gt <- simulate.genotypes(G, L, c, k, M)
  
  # Construct file name with variable values
  file_name <- paste("simulated_genotypes_G", G, "_L", L, "_c", c, "_k", k, "_M", M, ".csv", sep = "")
  
  # Save as a CSV file
  write.csv(gt, file = file_name, row.names = FALSE)
  
  # Print confirmation message
  cat("Simulated genotypes saved as:", file_name, "\n")
}



# Simulate genotypes and save as CSV
simulate_and_save_csv(G, L, c, k, M)
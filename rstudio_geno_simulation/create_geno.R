# Now you can use G, L, c, k, and M directly in your script
print(paste("G:", G))
print(paste("L:", L))
print(paste("c:", c))
print(paste("k:", k))
print(paste("M:", M))

# Example of using the arguments
example_rep <- rep(1:G, times = k)  # Example usage of 'rep' function
print(example_rep)

# Now you can use G, L, c, k, and M in your script

# Example of using the arguments
# Here you should check the usage of 'rep' function
example_rep <- rep(1:G, times = k)  # Example usage of 'rep' function
print(example_rep)

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
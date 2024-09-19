import numpy as np

# Set print options to ensure large matrices are fully displayed
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Function to perform matrix multiplication and print the inputs and outputs
def perform_matrix_multiplication(A, B):
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    
    # Perform matrix multiplication
    C = np.dot(A, B)
    
    print("Resultant Matrix:\n", C)
    print("="*50)  # Separator for clarity
    return C

# Define matrices
matrix_2x4 = np.random.randint(1, 10, (2, 4))
matrix_4x2 = np.random.randint(1, 10, (4, 2))
matrix_4x4 = np.random.randint(1, 10, (4, 4))
matrix_4x8 = np.random.randint(1, 10, (4, 8))
matrix_8x8 = np.random.randint(1, 10, (8, 8))
matrix_9x9 = np.random.randint(1, 10, (9, 9))
matrix_9x16 = np.random.randint(1, 10, (9, 16))
matrix_9x32 = np.random.randint(1, 10, (9, 32))
matrix_8x16 = np.random.randint(1, 10, (8, 16))
matrix_16x16 = np.random.randint(1, 10, (16, 16))

# Perform matrix multiplications
perform_matrix_multiplication(matrix_2x4, matrix_4x2)
perform_matrix_multiplication(matrix_2x4, matrix_4x8)
perform_matrix_multiplication(matrix_4x4, matrix_4x8)
perform_matrix_multiplication(matrix_4x8, matrix_8x8)
perform_matrix_multiplication(matrix_8x8, matrix_8x8)
perform_matrix_multiplication(matrix_8x8, matrix_8x16)
perform_matrix_multiplication(matrix_8x16, matrix_16x16)
perform_matrix_multiplication(matrix_9x9, matrix_9x9)
perform_matrix_multiplication(matrix_9x9, matrix_9x16)
perform_matrix_multiplication(matrix_9x9, matrix_9x32)

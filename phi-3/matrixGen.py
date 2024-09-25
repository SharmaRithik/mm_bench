import numpy as np
import os

# Create the folder 'fp16' if it doesn't exist
folder_name = 'fp16'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define matrix sizes
matrix_1TestA_size = (4, 4)
matrix_1TestB_size = (4, 6)
matrix_1A_size = (3072, 9216)
matrix_1B_size = (9216, 1)
matrix_2A_size = (3072, 3072)
matrix_2B_size = (3072, 1)
matrix_3A_size = (3072, 16384)
matrix_3B_size = (16384, 1)
matrix_4A_size = (8192, 3072)
matrix_4B_size = (3072, 1)
matrix_5A_size = (3072, 32064)
matrix_5B_size = (32064, 1)

# Generate random matrices with values between 0 and 9 for each layer
matrix_TestA = np.random.rand(*matrix_1TestA_size) * 9 
matrix_TestB = np.random.rand(*matrix_1TestB_size) * 9 

matrix_1A = np.random.rand(*matrix_1A_size) * 9 
matrix_1B = np.random.rand(*matrix_1B_size) * 9 

matrix_2A = np.random.rand(*matrix_2A_size) * 9 
matrix_2B = np.random.rand(*matrix_2B_size) * 9 

matrix_3A = np.random.rand(*matrix_3A_size) * 9 
matrix_3B = np.random.rand(*matrix_3B_size) * 9 

matrix_4A = np.random.rand(*matrix_4A_size) * 9 
matrix_4B = np.random.rand(*matrix_4B_size) * 9 

matrix_5A = np.random.rand(*matrix_5A_size) * 9 
matrix_5B = np.random.rand(*matrix_5B_size) * 9 

# Save matrices to text files inside the 'fp16' folder with 4 decimal places
np.savetxt(os.path.join(folder_name, 'layerTest_A.txt'), matrix_TestA, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layerTest_B.txt'), matrix_TestB, fmt='%.4f')

np.savetxt(os.path.join(folder_name, 'layer1_A.txt'), matrix_1A, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer1_B.txt'), matrix_1B, fmt='%.4f')

np.savetxt(os.path.join(folder_name, 'layer2_A.txt'), matrix_2A, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer2_B.txt'), matrix_2B, fmt='%.4f')

np.savetxt(os.path.join(folder_name, 'layer3_A.txt'), matrix_3A, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer3_B.txt'), matrix_3B, fmt='%.4f')

np.savetxt(os.path.join(folder_name, 'layer4_A.txt'), matrix_4A, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer4_B.txt'), matrix_4B, fmt='%.4f')

np.savetxt(os.path.join(folder_name, 'layer5_A.txt'), matrix_5A, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer5_B.txt'), matrix_5B, fmt='%.4f')

# Perform matrix multiplication
matrix_resultTest = np.dot(matrix_TestA, matrix_TestB)
matrix_result1 = np.dot(matrix_1A, matrix_1B)
matrix_result2 = np.dot(matrix_2A, matrix_2B)
matrix_result3 = np.dot(matrix_3A, matrix_3B)
matrix_result4 = np.dot(matrix_4A, matrix_4B)
matrix_result5 = np.dot(matrix_5A, matrix_5B)

# Save the result of the multiplication to a text file inside the 'fp16' folder
np.savetxt(os.path.join(folder_name, 'layerTest_Result.txt'), matrix_resultTest, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer1_Result.txt'), matrix_result1, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer2_Result.txt'), matrix_result2, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer3_Result.txt'), matrix_result3, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer4_Result.txt'), matrix_result4, fmt='%.4f')
np.savetxt(os.path.join(folder_name, 'layer5_Result.txt'), matrix_result5, fmt='%.4f')

print("Matrices and results saved successfully inside the 'fp16' folder.")


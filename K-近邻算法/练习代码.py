import numpy as np
random_array = np.random.rand(4, 4)
random_matrix_i = np.linalg.inv(random_array)
chengji=random_array*random_matrix_i
print(chengji)
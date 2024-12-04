import numpy as np

# file_path = "C:\\Users\\user\\generative-image-dynamics-main\\data\\flow\\Fleur_de_pommier_18_000.npy"
# file_path = "C:\\Users\\user\\generative-image-dynamics-main\\data\\flow\\Fleur_de_pommier_18_005.npy"

file_path = "C:\\Users\\user\\generative-image-dynamics-main\\data\\flow\\Lion_18_000.npy"
data = np.load(file_path)

print("Shape of data:", data.shape)

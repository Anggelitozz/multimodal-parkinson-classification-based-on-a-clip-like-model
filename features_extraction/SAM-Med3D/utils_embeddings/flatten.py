import os
import numpy as np

train_dir = "./embeddings"
test_dir  = "./embedding_test"

train_out = "./embeddings"
test_out  = "./embedding_test"

os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

def flatten_and_save(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            path_in = os.path.join(input_dir, file)
            path_out = os.path.join(output_dir, file)

            arr = np.load(path_in)
            arr_flat = arr.flatten()

            np.save(path_out, arr_flat)
            print(f"Procesado: {file} → {arr_flat.shape}")

# Procesar train y test
print("Procesando embeddings de entrenamiento...")
flatten_and_save(train_dir, train_out)

print("\nProcesando embeddings de test...")
flatten_and_save(test_dir, test_out)

print("\nTodos los embeddings han sido aplanados y guardados.")

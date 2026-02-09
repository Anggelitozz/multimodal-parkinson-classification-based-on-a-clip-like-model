import torch
import numpy as np

# Cargar el tensor desde el archivo .pt
emb = torch.load("/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_3104.pt")  

print("Forma del embedding:", emb.shape)

# Si está en GPU, lo pasamos a CPU primero
if emb.is_cuda:
    emb = emb.cpu()

# Convertir a numpy
emb_np = emb.numpy()

print("Forma en numpy:", emb_np.shape)
print("Tipo:", type(emb_np))

# Guardar como archivo .npy
np.save("T1_emb.npy", emb_np)
print("Guardado como T1_emb.npy")

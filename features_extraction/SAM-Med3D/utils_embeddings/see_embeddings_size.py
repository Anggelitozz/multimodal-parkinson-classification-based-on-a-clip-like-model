import numpy as np
import os
folder_train ="/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_test_spect"
folder="/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_spect"
for file in os.listdir(folder_train):
    emb = np.load(os.path.join(folder_train, file))
    print(file,": ",np.size(emb))
for file in os.listdir(folder):
    emb = np.load(os.path.join(folder, file))
    print(file, ": ",np.size(emb))


# emb = np.load("/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_test/3392.npy")
# print("tamaño desigual: ",np.size(emb))
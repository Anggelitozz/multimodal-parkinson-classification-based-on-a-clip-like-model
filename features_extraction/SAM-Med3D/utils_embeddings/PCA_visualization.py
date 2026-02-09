import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carpeta con embeddings
emb_dir = "/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_decoder_spect"

# Listas
X, y, ids = [], [], []

# Define aquí tus IDs de parkinson/control (0 = control, 1 = parkinson)
parkinson_ids = [3000, 3106, 3115, 3161, 3169, 3172, 3191, 3301, 3316, 3350, 3355, 3361, 3369, 3389, 3551, 3555, 3571, 3750, 3765, 3768,
3779, 3805, 3807, 3813, 3817, 3851, 3853, 3857, 4010, 4067, 3004, 3114, 3151, 3165, 3171, 3188, 3300, 3310, 3318, 3353,  
3357, 3368, 3370, 3390, 3554, 3563, 3572, 3756, 3767, 3769, 3804, 3806, 3812, 3816, 3850, 3852, 3854, 4004, 4018, 4139, 3001, 3105, 3127, 3166, 3181, 3354, 3392, 3556, 3577, 3815, 3818, 3832, 3867, 4001, 51632]
control_ids   = [3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321, 3323, 3328, 3364, 3380, 3552, 3564, 3585, 3753, 3764, 3780,
3800, 3822, 3830, 4005, 4019, 4025, 4030, 4035, 40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167,3178,3309, 3314, 3322, 
3327, 3360, 3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826, 3863, 4006, 4024, 4026, 4034, 4069, 41486,51731, 3104, 3112, 3157, 3160, 3320, 3358, 3565, 3569, 3570, 3759, 3803, 3811, 3855, 3859, 4032]


def load_all_embeddings(folder, ids_parkinson, ids_control):
    X, y = [], []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            id_ = int(file.split(".")[0])
            emb = np.load(os.path.join(folder, file))
            emb_flat = emb.flatten()

            if id_ in ids_parkinson:
                label = 1
            elif id_ in ids_control:
                label = 0
            else:
                continue

            X.append(emb_flat)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# === Cargar embeddings ===
X, y = load_all_embeddings(emb_dir, parkinson_ids, control_ids)
print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)

# === Reducir a 3D con PCA ===
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X)
print("PCA reducido:", X_3d.shape)

# === Visualización en 3D ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                     c=y, cmap="coolwarm", alpha=0.7)

ax.set_title("Visualización de Embeddings con PCA (3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
legend1 = ax.legend(*scatter.legend_elements(), title="Clases (0=Control, 1=Parkinson)")
ax.add_artist(legend1)

plt.tight_layout()
plt.savefig("pca_embeddings_3d.png", dpi=300)
print("✅ Gráfico guardado en pca_embeddings_3d.png")
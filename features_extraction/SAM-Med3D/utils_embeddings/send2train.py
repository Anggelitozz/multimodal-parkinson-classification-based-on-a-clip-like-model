# import os
# import shutil

# # === Carpetas ===
# src_dir = "./embeddings_test_spect"                 # donde están ahora todos tus .npy
# dst_dir = "./embeddings_spect"     # donde quieres mover los de train

# os.makedirs(dst_dir, exist_ok=True)

# # === Listas de IDs ===
# train_parkinson_ids = [
#     3000, 3106, 3115, 3161, 3169, 3172, 3191, 3301, 3316, 3350, 3355, 3361, 3369, 3389,
#     3551, 3555, 3571, 3750, 3765, 3768, 3779, 3805, 3807, 3813, 3817, 3851, 3853, 3857,
#     4010, 4067, 3004, 3114, 3151, 3165, 3171, 3188, 3300, 3310, 3318, 3353, 3357, 3368,
#     3370, 3390, 3554, 3563, 3572, 3756, 3767, 3769, 3804, 3806, 3812, 3816, 3850, 3852,
#     3854, 4004, 4018, 4139, 3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321,
#     3323, 3328, 3364, 3380, 3552, 3564, 3585, 3753, 3764, 3780, 3800, 3822, 3830, 4005,
#     4019, 4025, 4030, 4035, 40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167, 3178, 3309,
#     3314, 3322, 3327, 3360, 3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826,
#     3863, 4006, 4024, 4026, 4034, 4069, 41486, 51731
# ]

# train_control_ids   = [
#     3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321, 3323, 3328, 3364, 3380, 
#     3552, 3564, 3585, 3753, 3764, 3780, 3800, 3822, 3830, 4005, 4019, 4025, 4030, 4035, 
#     40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167, 3178, 3309, 3314, 3322, 3327, 3360, 
#     3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826, 3863, 4006, 4024, 4026, 
#     4034, 4069, 41486, 51731
# ]

# train_ids = set(train_parkinson_ids + train_control_ids)

# # === Mover archivos ===
# moved = 0
# for file in os.listdir(src_dir):
#     if file.endswith(".npy"):
#         try:
#             id_ = int(file.split(".")[0])
#         except ValueError:
#             print(f"⚠️ No se pudo extraer ID de {file}, lo salto...")
#             continue

#         if id_ in train_ids:
#             src_path = os.path.join(src_dir, file)
#             dst_path = os.path.join(dst_dir, file)
#             shutil.move(src_path, dst_path)
#             moved += 1
#             print(f"✅ Movido {file} → {dst_dir}")

# print(f"\nTotal de archivos movidos: {moved}")


import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# carpeta con todos los embeddings (train + test juntos)
embeddings_dir = "./embeddings_spect_all"

# Parkinson solo train
train_parkinson_ids = [
    3000, 3106, 3115, 3161, 3169, 3172, 3191, 3301, 3316, 3350, 3355, 3361, 3369, 3389,
    3551, 3555, 3571, 3750, 3765, 3768, 3779, 3805, 3807, 3813, 3817, 3851, 3853, 3857,
    4010, 4067, 3004, 3114, 3151, 3165, 3171, 3188, 3300, 3310, 3318, 3353, 3357, 3368,
    3370, 3390, 3554, 3563, 3572, 3756, 3767, 3769, 3804, 3806, 3812, 3816, 3850, 3852,
    3854, 4004, 4018, 4139
]

# Control solo train
train_control_ids = [
    3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321, 3323, 3328, 3364, 3380,
    3552, 3564, 3585, 3753, 3764, 3780, 3800, 3822, 3830, 4005, 4019, 4025, 4030, 4035,
    40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167, 3178, 3309, 3314, 3322, 3327, 3360,
    3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826, 3863, 4006, 4024, 4026,
    4034, 4069, 41486, 51731
]

# Test ya estaba bien definido
test_parkinson_ids  = [3001, 3105, 3127, 3166, 3181, 3354, 3392, 3556, 3577, 3815, 3818, 3832, 3867, 4001, 51632]
test_control_ids    = [3104, 3112, 3157, 3160, 3320, 3358, 3565, 3569, 3570, 3759, 3803, 3811, 3855, 3859, 4032]



def load_embeddings(folder, ids_parkinson, ids_control):
    X, y, ids = [], [], []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            id_ = int(file.split(".")[0])  # ID del archivo

            if id_ in ids_parkinson:
                label = 1
            elif id_ in ids_control:
                label = 0
            else:
                continue  # no pertenece ni a Parkinson ni a Control

            emb = np.load(os.path.join(folder, file))
            emb_flat = emb.flatten()

            X.append(emb_flat)
            y.append(label)
            ids.append(id_)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), ids


# Cargar solo lo que corresponde a train y test
X_train, y_train, ids_train = load_embeddings(embeddings_dir, train_parkinson_ids, train_control_ids)
X_test, y_test, ids_test    = load_embeddings(embeddings_dir, test_parkinson_ids, test_control_ids)


# Clasificadores
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Entrenar y evaluar
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n🔹 {name}")
    print(classification_report(y_test, y_pred, target_names=["Control", "Parkinson"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

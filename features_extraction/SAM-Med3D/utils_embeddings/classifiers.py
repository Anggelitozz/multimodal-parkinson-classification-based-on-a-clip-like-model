# import os
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# # carpetas con embeddings
# train_dir = "./first_try/embeddings"
# test_dir  = "./first_try/embeddings_test"

# # listas de IDs que ya tienes# Parkinson solo train
# train_parkinson_ids = [
#     3000, 3106, 3115, 3161, 3169, 3172, 3191, 3301, 3316, 3350, 3355, 3361, 3369, 3389,
#     3551, 3555, 3571, 3750, 3765, 3768, 3779, 3805, 3807, 3813, 3817, 3851, 3853, 3857,
#     4010, 4067, 3004, 3114, 3151, 3165, 3171, 3188, 3300, 3310, 3318, 3353, 3357, 3368,
#     3370, 3390, 3554, 3563, 3572, 3756, 3767, 3769, 3804, 3806, 3812, 3816, 3850, 3852,
#     3854, 4004, 4018, 4139
# ]

# # Control solo train
# train_control_ids = [
#     3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321, 3323, 3328, 3364, 3380,
#     3552, 3564, 3585, 3753, 3764, 3780, 3800, 3822, 3830, 4005, 4019, 4025, 4030, 4035,
#     40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167, 3178, 3309, 3314, 3322, 3327, 3360,
#     3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826, 3863, 4006, 4024, 4026,
#     4034, 4069, 41486, 51731
# ]

# # Test ya estaba bien definido
# test_parkinson_ids  = [3001, 3105, 3127, 3166, 3181, 3354, 3392, 3556, 3577, 3815, 3818, 3832, 3867, 4001, 51632]
# test_control_ids    = [3104, 3112, 3157, 3160, 3320, 3358, 3565, 3569, 3570, 3759, 3803, 3811, 3855, 3859, 4032]



# def load_embeddings(folder, ids_parkinson, ids_control):
#     X, y, ids = [], [], []
#     for file in os.listdir(folder):
#         if file.endswith(".npy"):
#             id_ = file.split(".")[0]  

#             emb = np.load(os.path.join(folder, file))
#             emb_flat = emb.flatten()

#             if int(id_) in ids_parkinson:
#                 print(id_,": PARKINSON\n")
#                 label = 1
#             elif int(id_) in ids_control:
#                 print(id_,": CONTROL\n")
#                 label = 0
#             else:
#                 continue 

#             X.append(emb_flat)
#             y.append(label)
#             ids.append(id_)
#     print(y)
#     emb = np.load(os.path.join(folder, file))
#     print(file, emb.shape, emb.size)
#     emb_flat = emb.flatten()

#     return np.array(X,dtype=np.float32), np.array(y,dtype=np.int64), ids

# X_train, y_train, ids_train = load_embeddings(train_dir, train_parkinson_ids, train_control_ids)
# X_test, y_test, ids_test    = load_embeddings(test_dir, test_parkinson_ids, test_control_ids)



# classifiers = {
#     "Logistic Regression": LogisticRegression(max_iter=500),
#     "SVM (RBF)": SVC(kernel="rbf", probability=True),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
# }

# for name, clf in classifiers.items():
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)[:, 1]

#     print(f"\n🔹 {name}")
#     print(classification_report(y_test, y_pred, target_names=["Control", "Parkinson"]))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("ROC AUC:", roc_auc_score(y_test, y_prob))



import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# carpeta con todos los embeddings (train + test juntos)
embeddings_dir = "/home/franklin_pupils/franklin_pupils/angel/SAM-Med3D-main/embeddings_mri_all_full_pd"

# # Parkinson solo train
# train_parkinson_ids = [
#     3000, 3106, 3115, 3161, 3169, 3172, 3191, 3301, 3316, 3350, 3355, 3361, 3369, 3389,
#     3551, 3555, 3571, 3750, 3765, 3768, 3779, 3805, 3807, 3813, 3817, 3851, 3853, 3857,
#     4010, 4067, 3004, 3114, 3151, 3165, 3171, 3188, 3300, 3310, 3318, 3353, 3357, 3368,
#     3370, 3390, 3554, 3563, 3572, 3756, 3767, 3769, 3804, 3806, 3812, 3816, 3850, 3852,
#     3854, 4004, 4018, 4139
# ]

# # Control solo train
# train_control_ids = [
#     3002, 3102, 3108, 3113, 3124, 3134, 3174, 3307, 3311, 3321, 3323, 3328, 3364, 3380,
#     3552, 3564, 3585, 3753, 3764, 3780, 3800, 3822, 3830, 4005, 4019, 4025, 4030, 4035,
#     40781, 50028, 3003, 3107, 3111, 3120, 3128, 3167, 3178, 3309, 3314, 3322, 3327, 3360,
#     3366, 3383, 3557, 3575, 3586, 3760, 3771, 3789, 3814, 3826, 3863, 4006, 4024, 4026,
#     4034, 4069, 41486, 51731
# ]

# # Test ya estaba bien definido
# test_parkinson_ids  = [3001, 3105, 3127, 3166, 3181, 3354, 3392, 3556, 3577, 3815, 3818, 3832, 3867, 4001, 51632]
# test_control_ids    = [3104, 3112, 3157, 3160, 3320, 3358, 3565, 3569, 3570, 3759, 3803, 3811, 3855, 3859, 4032]


control = ['3104', '3106', '3112', '3114', '3115', '3151', '3157', '3160', '3161', '3165', '3169', '3171', '3172', '3188', '3191', '3300', '3301', '3310', '3316', '3318', '3320', '3350', '3353', '3355', '3357', '3358', '3361', '3368', '3369', '3370', '3389', '3390', '3551', '3554', '3555', '3563', '3565', '3569', '3570', '3571', '3572', '3750', '3756', '3759', '3765', '3767', '3768', '3769', '3779', '3803', '3804', '3805', '3806', '3807', '3809', '3811', '3812', '3813', '3816', '3817', '3850', '3851', '3852', '3853', '3854', '3855', '3857', '3859', '4004', '4010', '4018', '4032', '4067', '4085', '4139']
pd = ['3102', '3105', '3107', '3108', '3111', '3113', '3116', '3118', '3119', '3120', '3122', '3123', '3124', '3125', '3126', '3127', '3128', '3129', '3130', '3131', '3132', '3134', '3150', '3154', '3166', '3167', '3168', '3173', '3174', '3175', '3176', '3178', '3179', '3181', '3182', '3184', '3185', '3190', '3305', '3307', '3308', '3309', '3311', '3314', '3321', '3322', '3323', '3325', '3327', '3328', '3332', '3352', '3354', '3359', '3360', '3364', '3365', '3366', '3367', '3371', '3372', '3373', '3374', '3375', '3376', '3377', '3378', '3380', '3383', '3385', '3386', '3387', '3392', '3552', '3556', '3557', '3558', '3559', '3564', '3567', '3574', '3575', '3577', '3584', '3585', '3586', '3587', '3588', '3589', '3591', '3592', '3593', '3752', '3753', '3757', '3758', '3760', '3762', '3763', '3764', '3770', '3771', '3775', '3776', '3777', '3778', '3780', '3781', '3787', '3788', '3789', '3800', '3802', '3808', '3814', '3815', '3818', '3819', '3822', '3823', '3824', '3825', '3826', '3827', '3828', '3829', '3830', '3831', '3832', '3833', '3834', '3835', '3837', '3838', '3863', '3866', '3867', '3868', '3869', '3870', '4001', '4005', '4006', '4011', '4012', '4013', '4019', '4020', '4021', '4022', '4024', '4025', '4026', '4027', '4029', '4030', '4033', '4034', '4035', '40366', '4037', '4038', '40533', '4065', '4069', '40781', '4080', '40800', '40806', '4081', '4082', '4083', '40882', '40893', '40916', '41184', '41289', '41293', '4135', '4136', '41486', '41488', '41664', '41829', '50028', '50319', '50485', '50901', '50983', '51632', '51731', '53060', '55395', '57037']

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
                print(id_)
                continue  # no pertenece ni a Parkinson ni a Control

            emb = np.load(os.path.join(folder, file))
            emb_flat = emb.flatten()

            X.append(emb_flat)
            y.append(label)
            ids.append(id_)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), ids


# Cargar solo lo que corresponde a train y test
print("---------------TRAIN-------------")
X_train, y_train, ids_train = load_embeddings(embeddings_dir, train_parkinson_ids, train_control_ids)
print("---------------TEST-------------")
X_test, y_test, ids_test    = load_embeddings(embeddings_dir, test_parkinson_ids, test_control_ids)



# Clasificadores
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

print(y_train)
print(y_test)
# Entrenar y evaluar
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f"\n🔹 {name}")
    print(classification_report(y_test, y_pred, target_names=["Control", "Parkinson"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

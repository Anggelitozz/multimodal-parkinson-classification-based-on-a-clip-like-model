import torch

# cargar el embebido guardado
emb = torch.load("embeddings_3104.pt")  # [1,C,D,H,W]

print("Forma del embedding:", emb.shape)   # debería dar algo como [1, 256, D, H, W]

# quitar batch
emb = emb.squeeze(0)  # → [C, D, H, W]

# ver las stats globales
print("Valor mínimo:", emb.min().item())
print("Valor máximo:", emb.max().item())
print("Media:", emb.mean().item())

# ver un slice de un canal específico
channel = 0
slice_idx = emb.shape[1] // 2
print("Slice de canal", channel, ":", emb[channel, slice_idx, :, :])

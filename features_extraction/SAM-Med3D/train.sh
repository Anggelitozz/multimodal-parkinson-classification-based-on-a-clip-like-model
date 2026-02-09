export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ruta del último checkpoint
CKPT_DIR="work_dir/ft_b2x1"
LATEST_CKPT="$CKPT_DIR/sam_model_latest.pth"

# Si existe un checkpoint anterior, reanuda desde ahí
if [ -f "$LATEST_CKPT" ]; then
    echo ">> Reanudando entrenamiento desde checkpoint: $LATEST_CKPT"
    python train.py \
        --batch_size 2 \
        --num_workers 4 \
        --task_name "ft_b2x1" \
        --checkpoint "$LATEST_CKPT" \
        --resume \
        --lr 8e-5
else
    echo ">> Iniciando entrenamiento desde cero con preentrenado ckpt/sam_med3d_turbo.pth"
    python train.py \
        --batch_size 2 \
        --num_workers 4 \
        --task_name "ft_b2x1" \
        --checkpoint "ckpt/sam_med3d_turbo.pth" \
        --lr 8e-5
fi
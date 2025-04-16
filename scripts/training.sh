python src/trainer.py \
    --wb-settings T D S \
    --model-name Style_AWB \
    --output_path "output" \
    --do-train \
    --do-eval \
    --training-dir datahub/training_data \
    --validation-dir datahub/validation_data \
    --epochs 200 \
    --batch-size 1 \
    --patch-size 64 \
    --patch-number 32 \
    --device 0 \
    --lr 1e-4 \
    -num-workers 4
    # --aug
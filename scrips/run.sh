python src/trainer.py \
    --wb-settings T D S \
    --model-name WB_model_Set1_new \
    --output_path "output" \
    --do-train \
    --training-dir datahub/cwcc_resize_5 \
    --epochs 200 \
    --batch-size 2 \
    --patch-size 64 \
    --patch-number 32 \
    --device 3 6 \
    --lr 1e-4 \
    --aug \

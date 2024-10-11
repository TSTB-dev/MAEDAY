# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors

python3 src/evaluate.py \
    --data_root ./data/mvtec_loco \
    --class_name breakfast_box \
    --img_size 224 \
    --num_masks 32 \
    --mask_ratio 0.75 \
    --weights_path weights/mae_finetune_vit_base_breakfast_box_mvtec_loco.pth \
    --gaussian_filter \
    --transform default \
    --seed 42 \
    --log_interval 1000 \
    --device cuda \

python3 src/evaluate.py \
    --data_root ./data/mvtec_loco \
    --class_name juice_bottle \
    --img_size 224 \
    --num_masks 32 \
    --mask_ratio 0.75 \
    --weights_path weights/mae_finetune_vit_base_juice_bottle_mvtec_loco.pth \
    --gaussian_filter \
    --transform default \
    --seed 42 \
    --log_interval 1000 \
    --device cuda \

python3 src/evaluate.py \
    --data_root ./data/mvtec_loco \
    --class_name pushpins \
    --img_size 224 \
    --num_masks 32 \
    --mask_ratio 0.75 \
    --weights_path weights/mae_finetune_vit_base_pushpins_mvtec_loco.pth \
    --gaussian_filter \
    --transform default \
    --seed 42 \
    --log_interval 1000 \
    --device cuda \

python3 src/evaluate.py \
    --data_root ./data/mvtec_loco \
    --class_name screw_bag \
    --img_size 224 \
    --num_masks 32 \
    --mask_ratio 0.75 \
    --weights_path weights/mae_finetune_vit_base_screw_bag_mvtec_loco.pth \
    --gaussian_filter \
    --transform default \
    --seed 42 \
    --log_interval 1000 \
    --device cuda \

python3 src/evaluate.py \
    --data_root ./data/mvtec_loco \
    --class_name splicing_connectors \
    --img_size 224 \
    --num_masks 32 \
    --mask_ratio 0.75 \
    --weights_path weights/mae_finetune_vit_base_splicing_connectors_mvtec_loco.pth \
    --gaussian_filter \
    --transform default \
    --seed 42 \
    --log_interval 1000 \
    --device cuda \
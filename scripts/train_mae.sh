# parser.add_argument("--data_root", type=str, default="/home/sakai/projects/MAEDAY/data/mvtec_ad")
# parser.add_argument("--class_name", type=str, default="bottle")
# parser.add_argument("--img_size", type=int, default=224)
# parser.add_argument("--split", type=str, default="train")
# parser.add_argument("--batch_size", type=int, default=32)

# parser.add_argument("--mask_ratio", type=float, default=0.75)
# parser.add_argument("--weights_path", type=str, default=None)  # path to the pre-trained weights

# parser.add_argument("--num_normal_samples", type=int, default=-1)  # -1: use all normal samples
# parser.add_argument("--lora", action="store_true", default=False)  # use LoRA, otherwise fine-tuning whole model
# parser.add_argument("--lora_dim", type=int, default=32)  
# parser.add_argument("--apply_loss_on_vis", action="store_true", default=False)  # apply loss on unmasked and masked patches
# parser.add_argument("--num_iters", type=int, default=50)
# parser.add_argument("--lr", type=float, default=0.01)
# parser.add_argument("--momentum", type=float, default=0.9)
# parser.add_argument("--weight_decay", type=float, default=0.05)
# parser.add_argument("--optimizer", type=str, default="sgd")
# parser.add_argument("--scheduler", type=str, default=None)

# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--num_workers", type=int, default=1)
# parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--log_interval", type=int, default=10)
# parser.add_argument("--save_dir", type=str, default="weights")

# carpet
# grid
# leather
# tile
# wood
# bottle
# cable
# capsule
# hazelnut
# metal_nut
# pill
# screw
# toothbrush
# transistor
# zipper

# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors

python3 src/train.py \
    --data_root ./data/mvtec_loco \
    --class_name breakfast_box \
    --img_size 224 \
    --split train \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --weights_path ./weights/mae_visualize_vit_base.pth \
    --training_scheme finetune \
    --finetune_mode decoder \
    --num_normal_samples -1 \
    --num_epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --seed 42 \
    --num_workers 0 \
    --device cuda \

python3 src/train.py \
    --data_root ./data/mvtec_loco \
    --class_name juice_bottle \
    --img_size 224 \
    --split train \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --weights_path ./weights/mae_visualize_vit_base.pth \
    --training_scheme finetune \
    --finetune_mode decoder \
    --num_normal_samples -1 \
    --num_epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --seed 42 \
    --num_workers 0 \
    --device cuda \

python3 src/train.py \
    --data_root ./data/mvtec_loco \
    --class_name pushpins \
    --img_size 224 \
    --split train \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --weights_path ./weights/mae_visualize_vit_base.pth \
    --training_scheme finetune \
    --finetune_mode decoder \
    --num_normal_samples -1 \
    --num_epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --seed 42 \
    --num_workers 0 \
    --device cuda \

python3 src/train.py \
    --data_root ./data/mvtec_loco \
    --class_name screw_bag \
    --img_size 224 \
    --split train \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --weights_path ./weights/mae_visualize_vit_base.pth \
    --training_scheme finetune \
    --finetune_mode decoder \
    --num_normal_samples -1 \
    --num_epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --seed 42 \
    --num_workers 0 \
    --device cuda \

python3 src/train.py \
    --data_root ./data/mvtec_loco \
    --class_name splicing_connectors \
    --img_size 224 \
    --split train \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --weights_path ./weights/mae_visualize_vit_base.pth \
    --training_scheme finetune \
    --finetune_mode decoder \
    --num_normal_samples -1 \
    --num_epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --seed 42 \
    --num_workers 0 \
    --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name grid \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name leather \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name tile \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name wood \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name bottle \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name cable \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name capsule \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name hazelnut \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name metal_nut \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name pill \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name screw \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name toothbrush \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name transistor \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \

# python3 src/train.py \
#     --data_root ./data/mvtec_ad \
#     --class_name zipper \
#     --img_size 224 \
#     --split train \
#     --batch_size 32 \
#     --mask_ratio 0.75 \
#     --weights_path ./weights/mae_visualize_vit_base.pth \
#     --training_scheme finetune \
#     --finetune_mode decoder \
#     --num_normal_samples -1 \
#     --num_epochs 100 \
#     --lr 0.01 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --seed 42 \
#     --num_workers 0 \
#     --device cuda \
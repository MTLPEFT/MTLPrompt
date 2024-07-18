# A15 0,1 Exp


# A15 0,1 Exp
RANDOM=$$
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000)) \
main.py --cfg './configs/mtlora/tiny_448/mtlora_plus_tiny_448_r64_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
--tasks semseg,normals,sal,human_parts --batch-size 32 --ckpt-freq=20 --epoch 300 \
--resume-backbone '../pretrained/swin_tiny_patch4_window7_224.pth' --use-checkpoint --disable_wandb


#B2 0 1 Exp
#RANDOM=$$
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 32 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth' --use-checkpoint -disable_wandb


#B2 2 3 Exp
#RANDOM=$$
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_plus_tiny_448_r16_scale4.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '../pretrained/swin_tiny_patch4_window7_224.pth' --use-checkpoint --disable_wandb




# B2 0 1 Exp
#RANDOM=$$
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r32_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth' --use-checkpoint

## B2 2 3 Exp
#RANDOM=$$
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth' --use-checkpoint


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r16_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 32 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth' --disable_wandb


#RANDOM=$$
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r32_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth'


#RANDOM=$$
#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2  --master_port=$((RANDOM%1000+12000)) \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/seungmin/pretrained/swin_tiny_patch4_window7_224.pth'


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r16_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/workspace/SMBaek/Experiments/pretrained/swin_tiny_patch4_window7_224_22k.pth'


#python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
#main.py --cfg './configs/mtlora/tiny_448/mtlora_tiny_448_r16_scale4_pertask.yaml' --pascal '/home/cvlab/datasets/PASCAL_MT' \
#--tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --epoch 300 \
#--resume-backbone '/home/cvlab/workspace/SMBaek/Experiments/pretrained/swin_tiny_patch4_window7_224_22k.pth' --disable_wandb


# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
#main.py --cfg configs/mtlora/tiny_448/<config>.yaml --pascal <path to pascal database> \
#--tasks semseg,normals,sal,human_parts --batch-size <batch size> --ckpt-freq=20 --epoch=<num epochs> \
#--resume-backbone <path to the weights of the chosen Swin variant>
#
#
#
#
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=3  --master_port=$((RANDOM%1000+12000))  \
#main.py --config_exp './configs/pascal/pascal_vitLp16_taskprompter.yml' --run_mode train #--trained_model  pretrained_mode0l_path
export CUDA_VISIBLE_DEVICES=0

DATA_PATH=/ssd1tb_00/byeonginjoung/dataset/sub001

python train.py --data_path $DATA_PATH --dataset mri --name_scene sub001
#python train.py --name_scene scene0758_00

nohup python train_batch.py resnet18 > ./resnet18.log 2>&1 &
nohup python train_batch.py resnet34 > ./resnet34.log 2>&1 &
nohup python train_batch.py resnet50 > ./resnet50.log 2>&1 &
CUDA_VISIBLE_DEVICES='1' nohup python train_batch.py resnet101 > ./resnet101.log 2>&1 &
CUDA_VISIBLE_DEVICES='1' nohup python train_batch.py resnet152 > ./resnet152.log 2>&1 &

NUM_GPUS=2

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS train_chatglm2.py 
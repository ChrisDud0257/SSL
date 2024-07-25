#!/usr/bin/env bash
### ESRGANSSL
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/ESRGANSSL/test_MultiGTStep3V1_01_x4.yml




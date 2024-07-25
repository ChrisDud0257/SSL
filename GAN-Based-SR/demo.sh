#!/usr/bin/env bash
### ESRGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/ESRGANSSL/train_ESRGANSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/ESRGANSSL/test_ESRGANSSL_bicubic_x4.yml



### RankSRGANPISSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/RankSRGANPISSL/train_RankSRGANPISSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/RankSRGANPISSL/train_RankSRGANPISSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/RankSRGANPISSL/test_RankSRGANPISSL_bicubic_x4.yml



### SPSRSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/SPSRSSL/train_SPSRSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/SPSRSSL/train_SPSRSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/SPSRSSL/test_SPSRSSL_bicubic_x4.yml



### BebyGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/BebyGANSSL/train_BebyGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/BebyGANSSL/train_BebyGANSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/BebyGANSSL/test_BebyGANSSL_bicubic_x4.yml




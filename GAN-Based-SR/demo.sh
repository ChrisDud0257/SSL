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



### LDLSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/LDLSSL/train_LDLSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/LDLSSL/train_LDLSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/LDLSSL/test_LDLSSL_bicubic_x4.yml



### SwinIRGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/SwinIRGANSSL/train_SwinIRGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/SwinIRGANSSL/train_SwinIRGANSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/SwinIRGANSSL/test_SwinIRGANSSL_bicubic_x4.yml



### SwinIRGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/ELANGANSSL/train_ELANGANSSL_bicubic_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/ELANGANSSL/train_ELANGANSSL_bicubic_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/ELANGANSSL/test_ELANGANSSL_bicubic_x4.yml


### BSRGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=5432 ./train_BSGRAN/main_train_SSL.py --opt ./train_BSGRAN/options/BSRGAN/train_BSRGANSSL_DF2K_OST_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/BSRGANSSL/test_BSRGANSSL_DF2K_OST_x4.yml



### RealESRGANSSL
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt ./options/train/RealESRGANSSL/train_RealESRGANSSL_x4.yml --launcher pytorch --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/RealESRGANSSL/test_RealESRGANSSL_x4.yml



### SwinIRGANSSL_BSRGAN
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=5432 ./train_BSGRAN/main_train_SSL.py --opt ./train_BSGRAN/options/BSRGAN/train_SwinIRGANSSL_BSRGAN_DF2K_OST_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/SwinIRGANSSL_BSRGAN/test_SwinIRGANSSL_BSRGAN_DF2K_OST_x4.yml



### ELANGANSSL_BSRGAN
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=5432 ./train_BSGRAN/main_train_SSL.py --opt ./train_BSGRAN/options/BSRGAN/train_ELANGANSSL_BSRGAN_DF2K_OST_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/ELANGANSSL_BSRGAN/test_ELANGANSSL_BSRGAN_DF2K_OST_x4.yml




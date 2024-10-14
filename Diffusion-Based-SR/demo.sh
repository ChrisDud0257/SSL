#python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 6,7 --name NAME --scale_lr False

#python main.py --train --base ./configs/mystableSRNew/v2-finetune_text_T_512.yaml --gpus 0,1,2,3 --name MystableSRNew --scale_lr False

#CUDA_VISIBLE_DEVICES=0 \
#python ./scripts/gt_input_output.py

python main.py --train --base configs/StableSRISSLStage1/StableSRISSLStage1_ks25_kc9_h0.004_1000.yml --gpus 0, --name NAME --scale_lr False
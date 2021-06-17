python inference.py \
"/Users/socretlee/Google Drive/3-resources/dataset/sk-object-masks/test/*" \
configs/ssd300_mobilenetv2.json \
/Users/socretlee/Google\ Drive/1-projects/ssd300_mobilenetv2_sk15_2_480x640_500/cp_38_loss-3.28.h5 \
--label_maps=/Users/socretlee/CodingDrive/other/skol-khmer-synthetic-dataset/output/label_maps.txt \
--confidence_threshold=0.9 \
--num_predictions=10
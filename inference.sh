python inference.py \
"/Users/socretlee/CodingDrive/other/skol-khmer-synthetic-dataset/output/images/*" \
configs/ssd300_mobilenetv2.json \
/Users/socretlee/Google\ Drive/1-projects/ssd300_mobilenetv2_sk15_2_480x640_500/cp_13_loss-5.30.h5 \
--label_maps=/Users/socretlee/CodingDrive/other/skol-khmer-synthetic-dataset/output/label_maps.txt \
--confidence_threshold=0.9 \
--num_predictions=10
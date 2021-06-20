python inference.py \
"data/pascal-voc-2007/images/*" \
configs/ssd300_vgg16_pascal-voc-2007.json \
/Users/socretlee/Google\ Drive/1-projects/ssd300_vgg16_pascal-voc-2007_trainval/cp_166_loss-5.24_valloss-5.99.h5 \
--label_maps=/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/label_maps.txt \
--confidence_threshold=0.8 \
--num_predictions=100
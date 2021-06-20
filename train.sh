python train.py \
configs/ssd300_vgg16_pascal-voc-2007.json \
/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/images \
/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/labels \
--training_split=/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/train.txt \
--validation_split=/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/val.txt \
--label_maps=/Users/socretlee/CodingDrive/other/object-detection-in-keras/data/pascal-voc-2007/label_maps.txt \
--learning_rate=0.001 \
--epochs=100 \
--batch_size=32 \
--shuffle=False \
--augment=True \
--output_dir=output/ssd300_vgg16_pascal-voc-2007
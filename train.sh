python train.py \
configs/ssd300_vgg16.json \
/Users/socretlee/Downloads/voc-07-12/label_maps.txt \
/Users/socretlee/Downloads/voc-07-12/images \
/Users/socretlee/Downloads/voc-07-12/annotations \
/Users/socretlee/Downloads/voc-07-12/trainval.txt \
--learning_rate=10e-3 \
--epochs=1 \
--batch_size=32 \
--checkpoint_frequency=1
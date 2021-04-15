python train.py \
configs/qssd300_vgg16.json \
output/midv500/images \
output/midv500/labels \
--training_split=output/midv500/train.txt \
--validation_split=output/midv500/val.txt \
--label_maps=output/label_maps.txt \
--learning_rate=0.001 \
--epochs=1 \
--batch_size=32 \
--shuffle=True \
--output_dir=output/qssd300_vgg16_midv500
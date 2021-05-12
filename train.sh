python train.py \
configs/ssd300_mobilenetv2.json \
output/midv500/images \
output/midv500/labels \
--training_split=output/midv500/train.txt \
--validation_split=output/midv500/val.txt \
--label_maps=output/label_maps.txt \
--learning_rate=0.001 \
--epochs=100 \
--batch_size=32 \
--shuffle=False \
--output_dir=output/ssd300_mobilenetv2_midv500_test
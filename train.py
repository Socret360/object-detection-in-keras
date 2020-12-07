import json
from tensorflow.keras.optimizers import Adam, SGD
from models import SSD300_VGG16_ORIGINAL
from losses import SSD_LOSS
from data_generators import SSD_VOC_DATA_GENERATOR

if __name__ == "__main__":
    with open("configs/ssd300_vgg16_original.json") as config_file:
        config = json.load(config_file)

    SSD300_VGG16_ORIGINAL(config)

    # training config
    # train_data_generator = SSD_VOC_DATA_GENERATOR(
    #     training_set_path,
    #     data_dir,
    #     input_shape,
    #     batch_size,
    #     num_classes,
    #     shuffle,
    #     aspect_ratios,
    #     variances,
    #     extra_box_for_ar_1
    # )

    # model = SSD300_VGG16_ORIGINAL(
    #     batch_size=batch_size,
    #     input_shape=input_shape,
    #     l2_reg=l2_reg,
    #     extra_box_for_ar_1=extra_box_for_ar_1,
    #     normalize_coords=normalize_coords,
    #     variances=variances,
    #     num_classes=num_classes,
    #     default_boxes_config=default_boxes_config,
    # )
    # sgd_optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    # ssd_loss = SSD_LOSS()
    # model.compile(optimzer=sgd_optimizer, loss=ssd_loss.compute)

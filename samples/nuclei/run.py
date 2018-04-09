import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
	
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()

import ast,os,cv2
import pandas as pd
import numpy as np

_WORKPATH = r'C:\Users\ahmad\Kaggle\data-science-bowl-2018'
_TRAINCSV = os.path.join(_WORKPATH,'Train.csv')
_TESTCSV = os.path.join(_WORKPATH,'Test.csv')

cv2.setUseOptimized(True)

class NucleiDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def loadimg(self, path, color=cv2.IMREAD_COLOR, size=None):
        img = cv2.imread(path, color)
        if size:
            img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
        return img

    def load_nuclei(self, count, height, width, aug=False):

        self.add_class("shapes", 1, "nuclei")
        traindf = pd.read_csv(_TRAINCSV)

        for i in range(len(traindf['img_id'])):
            self.add_image("shapes", image_id=str(i), path=traindf['image_path'].loc[i], width=width, height=height, maskpath=traindf['mask_dir'].loc[i])
            if 'augs_path' in traindf.columns and aug:
                for j,augpath in enumerate(ast.literal_eval(traindf['augs_path'].loc[i])):
                    self.add_image("shapes", image_id=str(i)+"_"+str(j), path=augpath, width=width, height=height, maskpath=ast.literal_eval(traindf['augs_path'].loc[i])[j])


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return "nuclei"
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id, size=None):
        """Generate instance masks for shapes of the given image ID.
        """
        class_ids = []
        info = self.image_info[image_id]
        rawmask= info['maskpath']
        count = len(rawmask) if  ".png" in rawmask else len(os.listdir(rawmask))
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        if ".png" in rawmask:
            _masktmp = self.loadimg(rawmask, cv2.IMREAD_COLOR, size=(info['height'], info['width']))
            mask= _masktmp[:, :, np.newaxis]
        else:
            for i, path in enumerate(next(os.walk(rawmask))[2]):
                _maskpath = os.path.join(rawmask, path)
                _masktmp = self.loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size=(info['height'], info['width']))
                mask[:, :, i:i] = _masktmp[:, :, np.newaxis]
        for i in range(count):
            class_ids.append(1)
        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)
		
# Training dataset
dataset_train = NucleiDataset()
dataset_train.load_nuclei(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset()
dataset_val.load_nuclei(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()
print(type(dataset_val))

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
						  
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
	
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')
print("DONE")

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
print("DONE")

model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)

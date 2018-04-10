import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import tqdm
import matplotlib.pyplot as plt
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
from skimage.morphology import label
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

res = 512
cut = 0.9
ytrue_predict=[]
ytrue_nameid=[]

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
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

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

import ast, os, cv2, skimage.io, skimage.color, skimage.transform
import pandas as pd
import numpy as np
from mrcnn import utils

_MODEL = "RCNN"
_WORKPATH = "../../../"
_TRAINCSV = os.path.join(_WORKPATH, _MODEL+'_Train.csv')
_TESTCSV = os.path.join(_WORKPATH, 'Test.csv')

cv2.setUseOptimized(True)


class NucleiDataset(utils.Dataset):

    def loadimg(self, path, color=cv2.IMREAD_COLOR, size=None):
        img = cv2.imread(path, color)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def load_nuclei(self, height, width, aug=False, train=False):
    #Pre-augment data disable
        self.add_class("shapes", 1, "nuclei")
        if train:
            traindf = pd.read_csv(_TRAINCSV)

            for i in range(len(traindf['img_id'])):
                self.add_image("shapes", image_id=str(i), path=traindf['image_path'].loc[i], width=traindf['img_width'].loc[i], height=traindf['img_height'].loc[i])
                if 'augs_path' in traindf.columns and aug:
                    for j, augpath in enumerate(ast.literal_eval(traindf['augs_path'].loc[i])):
                        self.add_image("shapes", image_id=str(i) + "_" + str(j), path=augpath, width=width, height=height,
                                       maskpath=ast.literal_eval(traindf['augs_path'].loc[i])[j])
        else:
            traindf = pd.read_csv(_TESTCSV)

            for i in range(len(traindf['img_id'])):
                self.add_image("shapes", image_id=str(i), path=traindf['image_path'].loc[i], width=traindf['img_width'].loc[i], height=traindf['img_height'].loc[i], name_id=str(traindf['img_name'].loc[i]).replace('.png',''))


    def image_reference(self, image_id):

        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return "nuclei"
        else:
            super(self.__class__).image_reference(image_id)

    def load_mask(self, image_id, size=None):

        class_ids = []
        info = self.image_info[image_id]
        rawmask = info['maskpath']
        count = len(rawmask) if ".png" in rawmask else len(os.listdir(rawmask))
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        if ".png" in rawmask:
            _masktmp = self.loadimg(rawmask, cv2.IMREAD_COLOR, size=None)
            mask = _masktmp[:, :, np.newaxis]
        else:
            for i, path in enumerate(next(os.walk(rawmask))[2]):
                _maskpath = os.path.join(rawmask, path)
                _masktmp = self.loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size=None)
                mask[:, :, i:i + 1] = _masktmp[:, :, np.newaxis]

        for i in range(count):
            class_ids.append(1)
        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)

dataset_val = NucleiDataset()
dataset_val.load_nuclei(config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=cut):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MAX_INSTANCES = 500
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)


model_path = os.path.join(MODEL_DIR, "mask_rcnn_256r100e8b.h5")

assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

for i,image_id in tqdm.tqdm(enumerate(dataset_val.image_ids), total=len(dataset_val.image_ids), unit='images'):
    info = dataset_val.image_info[image_id]
    print(info['name_id'])
    imagex = dataset_val.load_image(image_id)
    image = cv2.resize(imagex, (res, res), interpolation=cv2.INTER_CUBIC)
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])
    p, q, r = det_masks.shape
    for k, j in enumerate(det_masks):
        newblank = np.zeros([q, r], dtype=np.uint8)
        ntmpmask = j
        if not k:
            nmask = newblank
        else:
            nmask = np.maximum(nmask, ntmpmask)

    rmask = cv2.resize(nmask, (int(info['width']), int(info['height'])), interpolation=cv2.INTER_CUBIC)
    rle = list(prob_to_rles(rmask))
    ytrue_predict.extend(rle)
    ytrue_nameid.extend([str(info['name_id'])] * len(rle))
sub = pd.DataFrame()
sub['ImageId'] = ytrue_nameid
sub['EncodedPixels'] = pd.Series(ytrue_predict).apply(lambda x: ' '.join(str(y) for y in x))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
print('Submission output to: sub-{}.csv'.format(timestamp))
sub.to_csv((os.path.join(MODEL_DIR,"sub-{}_{} r_{} morph-{}.csv".format(_MODEL,res,str(cut).replace('.',''),timestamp))), index=False)
print("DONE")





def testz():
    #image_id = random.choice(dataset_val.image_ids)
    image_id = 32
    print("Total test data : {}".format(len(dataset_val.image_ids)))
    #TEST
    imagex = dataset_val.load_image(image_id)
    image = cv2.resize(imagex,(512,512),interpolation=cv2.INTER_CUBIC)
    #image = skimage.transform.resize(imagex,(512,512), mode= 'constant', preserve_range=True)
    info = dataset_val.image_info[image_id]
    print("ID : {} | num : {}".format((info['name_id']),image_id))
    print("Res : {}x{}".format(info['height'],info['width']))
    print("")
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    try:
        det_count = np.where(det_class_ids == 0)[0][0]
        print(np.where((det_class_ids == 0))[0])
    except:
        det_count = 100
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(det_count, np.array(dataset_val.class_names)[det_class_ids]))

    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])
    finalresult=np.expand_dims(modellib.mold_image(image, inference_config), 0)#not sure for what
    print("image : {} | mask : {}".format(imagex.shape,det_masks.shape))

    #stacking all mask
    deltax = int(info['width'])-256
    deltay = int(info['height'])-256
    p, q, r = det_masks.shape
    for k,j in enumerate(det_masks):
        newblank = np.zeros([q,r], dtype=np.uint8)
        ntmpmask = j
        if not k:
            nmask = newblank
        else:
            nmask = np.maximum(nmask, ntmpmask)

    rmask = cv2.resize(nmask,(int(info['width']),int(info['height'])),interpolation=cv2.INTER_CUBIC)
    #rmask = skimage.transform.resize(nmask,(int(info['height']), int(info['width'])), mode= 'constant', preserve_range=True)


    plt.imsave(fname= os.path.join(ROOT_DIR, "maskresize.png"), arr = rmask)
    plt.imsave(fname= os.path.join(ROOT_DIR, "ori.png"), arr = imagex)
    plt.imsave(fname= os.path.join(ROOT_DIR, "mask.png"), arr = nmask)


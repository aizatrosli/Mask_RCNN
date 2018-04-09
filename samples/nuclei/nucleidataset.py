import ast, os, cv2, skimage.io, skimage.color, skimage.transform
import pandas as pd
import numpy as np
from mrcnn import utils

_MODEL = "RCNN"
_WORKPATH = r'C:\Users\ahmad\Kaggle\data-science-bowl-2018'
_TRAINCSV = os.path.join(_WORKPATH, 'Train.csv')
_TESTCSV = os.path.join(_WORKPATH, _MODEL+'Test.csv')

cv2.setUseOptimized(True)


class NucleiDataset(utils.Dataset):

    def loadimg(self, path, color=cv2.IMREAD_COLOR, size=None):
        img = cv2.imread(path, color)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = skimage.transform.resize(image, (info['height'], info['width']),mode='constant')
        return image

    def load_nuclei(self, height, width, aug=False):
    #Pre-augment data disable
        self.add_class("shapes", 1, "nuclei")
        traindf = pd.read_csv(_TRAINCSV)

        for i in range(len(traindf['img_id'])):
            self.add_image("shapes", image_id=str(i), path=traindf['image_path'].loc[i], width=width, height=height,
                           maskpath=traindf['mask_dir'].loc[i])
            if 'augs_path' in traindf.columns and aug:
                for j, augpath in enumerate(ast.literal_eval(traindf['augs_path'].loc[i])):
                    self.add_image("shapes", image_id=str(i) + "_" + str(j), path=augpath, width=width, height=height,
                                   maskpath=ast.literal_eval(traindf['augs_path'].loc[i])[j])

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
            _masktmp = self.loadimg(rawmask, cv2.IMREAD_COLOR, size=(info['height'], info['width']))
            mask = _masktmp[:, :, np.newaxis]
        else:
            for i, path in enumerate(next(os.walk(rawmask))[2]):
                _maskpath = os.path.join(rawmask, path)
                _masktmp = self.loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size=(info['height'], info['width']))
                mask[:, :, i:i + 1] = _masktmp[:, :, np.newaxis]

        for i in range(count):
            class_ids.append(1)
        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)
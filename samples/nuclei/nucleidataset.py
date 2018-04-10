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

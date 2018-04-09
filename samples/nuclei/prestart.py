import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform

import cv2, os, sys, tqdm, random, ast, h5py, pickle

#use optimised settings
cv2.setUseOptimized(True)
#config parameters
_PICKLEGEN = False
_

num_gpus = 1
_MODEL = "RCNN"
#_MODEL = "unet"
_HARDIMG = 'f952cc65376009cfad8249e53b9b2c0daaa3553e897096337d143c625c2df886'

#set scale train images
_IMGHEIGHT = 256
_IMGWIDTH = 256
_IMGCOLOR = 3
_IMGTYPE = '.png'
#getting workspace path
_WORKPATH = os.getcwd()
_TESTPATH = os.path.join(_WORKPATH,"stage1_test")
_TRAINPATH = os.path.join(_WORKPATH,"stage1_train")

# Set seed values
seed = 42
random.seed = seed
np.random.seed(seed=seed)

print("Workspace : {}".format(_WORKPATH))
print("Train : {}".format(_TRAINPATH))
print("Test : {}".format(_TESTPATH))

def loadimg(path, color=cv2.IMREAD_COLOR, size=None):
    img = cv2.imread(path, color)
    if size:
        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    return img

def loadmaskaug(dirs, size=None):
    for i,path in enumerate(next(os.walk(dirs))[2]):
        _maskpath = os.path.join(dirs, path)
        _masktmp = loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size)
        #stacking mask image
        if not i: mask = _masktmp
        else: mask = np.maximum(mask, _masktmp)
    return mask

def loadmask(dirs, size=None):
    count = len(os.listdir(dirs))
    mask = np.zeros([_IMGHEIGHT, _IMGWIDTH, count], dtype=np.uint8)
    for i,path in enumerate(next(os.walk(dirs))[2]):
        _maskpath = os.path.join(dirs, path)
        _masktmp = loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size)
        mask[:, :, i:i] = _masktmp[:,:,np.newaxis]
    return mask

def createdt(path):
    tmp = []
    for i, _dirpath in enumerate(os.listdir(path)):
        _imgdir = os.path.join(path,_dirpath,"images")
        _maskdir = None
        if "masks" in os.listdir(os.path.join(path,_dirpath)):
            _maskdir = os.path.join(path,_dirpath,"masks")
            _nummasks = len(os.listdir(_maskdir))
        _imgname = os.listdir(_imgdir)[0] if len(os.listdir(_imgdir)) < 2 else "Multiple Image Source"
        _imgpath = os.path.join(_imgdir,_imgname)
        _imgshape = loadimg(_imgpath).shape
        tmp.append([i, _imgname, _imgshape[0], _imgshape[1], _imgshape[0]/_imgshape[1], _imgshape[2], _nummasks, _imgpath, _maskdir] ) if _maskdir else tmp.append([i, _imgname, _imgshape[0], _imgshape[1], _imgshape[0]/_imgshape[1], _imgshape[2], _imgpath])

    dt_df = pd.DataFrame(tmp, columns= ['img_id', 'img_name', 'img_height', 'img_width', 'img_ratio', 'num_channels', 'image_path']) if len(tmp[0]) < 8 else pd.DataFrame(tmp, columns= ['img_id', 'img_name', 'img_height', 'img_width', 'img_ratio', 'num_channels', 'num_masks', 'image_path', 'mask_dir'])
    return dt_df



def augdata(img, mask, resize_rate =0.85,angle = 30):
    flip = random.randint(0, 1)
    size = img.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angle
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    img = transform.warp(img, inverse_map=afine_tf,mode='constant')
    mask = transform.warp(mask, inverse_map=afine_tf,mode='constant')
    # Randomly corpping image frame
    img = img[w_s:w_s+size,h_s:h_s+size,:]
    mask = mask[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        img = img[:,::-1,:]
        mask = mask[:,::-1]
    img = transform.resize(img,(256,256),mode='edge')
    mask = transform.resize(mask,(256,256),mode='edge')
    return img, mask


def loadalldata(aug = 0, plotsample = False,size=(_IMGHEIGHT,_IMGWIDTH)):
    x_train, y_train, x_test = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    if aug:
        print('Image Augmentation is Enabled')
        print('Parsing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df), unit='images'):
        img = loadimg(train_df['image_path'].loc[i], size=size)
        mask = loadmaskaug(train_df['mask_dir'].loc[i], size=size) if _MODEL == "unet" else loadmask(train_df['mask_dir'].loc[i], size=size)
        x_train.append(img)
        y_train.append(mask)
        if aug:
            if not 'augs_path' in train_df.columns:
                train_df['augs_path'] = ''
                train_df['augsmasks_path'] = ''
            _AUGPATHCOL = []
            _AUGMASKCOL = []
            for j in range(0, int(aug)):
                augimg, augmask = augdata(img, mask)
                _AUGPATH = os.path.join(_TRAINPATH,str(train_df['img_name'].loc[i]).replace(".png",""),"augs", str(train_df['img_name'].loc[i]).replace(".png","_{}.png".format(j)))
                _AUGMASKPATH = os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks", str(train_df['img_name'].loc[i]).replace(".png","_{}.png".format(j)))
                if not os.path.exists(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""),"augs")): os.makedirs(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""),"augs"))
                if not os.path.exists(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks")): os.makedirs(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks"))
                if str(train_df['img_name'].loc[i].replace(".png","")) == _HARDIMG and plotsample:
                    print('Checking hardest image augmentation')
                    plt.subplot(221)
                    plt.imshow(img)
                    plt.subplot(222)
                    plt.imshow(mask)
                    plt.subplot(223)
                    plt.imshow(augimg)
                    plt.subplot(224)
                    plt.imshow(augmask)
                plt.imsave(fname= _AUGPATH, arr = augimg.reshape(img.shape))
                plt.imsave(fname=_AUGMASKPATH, arr = augmask.reshape(_IMGHEIGHT,_IMGWIDTH))
                _AUGPATHCOL.append(str(_AUGPATH))
                _AUGMASKCOL.append(str(_AUGMASKPATH))
                x_train.append(augimg)
                y_train.append(augmask)
            train_df['augs_path'].loc[i] = _AUGPATHCOL
            train_df['augsmasks_path'].loc[i] = _AUGMASKCOL
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df), unit='images'):
        img = loadimg(test_df['image_path'].loc[i], size=size)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    print ("Train Images : {} and expected : {}".format(len(x_train),len(train_df)*(int(aug)+1) if aug else len(train_df)))
    x_train = np.array(x_train, dtype=np.uint8)
    #y_train = np.array(y_train, dtype=np.bool)[:,:,:,np.newaxis] if _MODEL == "unet" else np.array(y_train)
    if _MODEL == "unet": y_train = np.array(y_train, dtype=np.bool)[:, :, :, np.newaxis]
    x_test = np.array(x_test)
    return x_train, y_train, x_test


train_df = createdt(_TRAINPATH)
test_df = createdt(_TESTPATH)
x_train, y_train, x_test = loadalldata()

if _PICKLEGEN:
    pickle_out = open("dataset.pkl","wb")
    dictdata = {"trainimg" : x_train, "trainmask" : y_train, "testimg" : x_test}
    pickle.dump(dictdata,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
    pickle_out.close()


test_df.to_csv(os.path.join(_WORKPATH, "Test.csv"),header=True,index=False)
train_df.to_csv(os.path.join(_WORKPATH,_MODEL+"_Train.csv"),header=True,index=False)


if _PICKLEGEN:
    pickle_in = open("dataset.pkl","rb")
    datain = pickle.load(pickle_in)
    kappa = datain
    pickle_in.close()
    print(len(kappa))

#Testing csv if working or not
train_df_csv = pd.read_csv(os.path.join(_WORKPATH,_MODEL+"_Train.csv"))
test_df_csv = pd.read_csv(os.path.join(_WORKPATH,"Test"))
print('train_df_CSV:')
print(train_df_csv.describe())
print('')
print('test_df_CSV:')
print(test_df_csv.describe())
import os
import os.path as osp
import glob
from pathlib import Path
import cv2
import numpy as np
import json

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

def _np_types(*names: str):
    return tuple(getattr(np, name) for name in names if hasattr(np, name))


NP_BOOL_TYPES = _np_types("bool_", "bool8")
NP_FLOAT_TYPES = _np_types("float16", "float32", "float64", "float_")
NP_INT_TYPES = _np_types("int8", "int16", "int32", "int64", "int_", "uint8", "uint16", "uint32", "uint64", "uint")

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)

def find_all_imgs(img_dir, abs_path=False):
    imglist = list()
    for filep in glob.glob(osp.join(img_dir, "*")):
        filename = osp.basename(filep)
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(filep)
        else:
            imglist.append(filename)
    return imglist

imread = lambda imgpath, read_type=cv2.IMREAD_COLOR: cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
# def imread(imgpath, read_type=cv2.IMREAD_COLOR):
#     img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
#     return img

def imwrite(img_path, img, ext='.png'):
    suffix = Path(img_path).suffix
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    cv2.imencode(ext, img)[1].tofile(img_path)

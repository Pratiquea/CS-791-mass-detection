#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
# import sys




# import bisect
# from functools import wraps

# from torch.utils.data.dataset import ConcatDataset as torchConcatDataset
# from torch.utils.data.dataset import Dataset as torchDataset


# class ConcatDataset(torchConcatDataset):
#     def __init__(self, datasets):
#         super(ConcatDataset, self).__init__(datasets)
#         if hasattr(self.datasets[0], "input_dim"):
#             self._input_dim = self.datasets[0].input_dim
#             self.input_dim = self.datasets[0].input_dim

#     def pull_item(self, idx):
#         if idx < 0:
#             if -idx > len(self):
#                 raise ValueError(
#                     "absolute value of index should not exceed dataset length"
#                 )
#             idx = len(self) + idx
#         dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#         if dataset_idx == 0:
#             sample_idx = idx
#         else:
#             sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
#         return self.datasets[dataset_idx].pull_item(sample_idx)


# class MixConcatDataset(torchConcatDataset):
#     def __init__(self, datasets):
#         super(MixConcatDataset, self).__init__(datasets)
#         if hasattr(self.datasets[0], "input_dim"):
#             self._input_dim = self.datasets[0].input_dim
#             self.input_dim = self.datasets[0].input_dim

#     def __getitem__(self, index):

#         if not isinstance(index, int):
#             idx = index[1]
#         if idx < 0:
#             if -idx > len(self):
#                 raise ValueError(
#                     "absolute value of index should not exceed dataset length"
#                 )
#             idx = len(self) + idx
#         dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#         if dataset_idx == 0:
#             sample_idx = idx
#         else:
#             sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
#         if not isinstance(index, int):
#             index = (index[0], sample_idx, index[2])

#         return self.datasets[dataset_idx][index]


# class Dataset(torchDataset):
#     """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
#     that enables on the fly resizing of the ``input_dim``.

#     Args:
#         input_dimension (tuple): (width,height) tuple with default dimensions of the network
#     """

#     def __init__(self, input_dimension, mosaic=True):
#         super().__init__()
#         self.__input_dim = input_dimension[:2]
#         self.enable_mosaic = mosaic

#     @property
#     def input_dim(self):
#         """
#         Dimension that can be used by transforms to set the correct image size, etc.
#         This allows transforms to have a single source of truth
#         for the input dimension of the network.

#         Return:
#             list: Tuple containing the current width,height
#         """
#         if hasattr(self, "_input_dim"):
#             return self._input_dim
#         return self.__input_dim

#     @staticmethod
#     def mosaic_getitem(getitem_fn):
#         """
#         Decorator method that needs to be used around the ``__getitem__`` method. |br|
#         This decorator enables the closing mosaic

#         Example:
#             >>> class CustomSet(ln.data.Dataset):
#             ...     def __len__(self):
#             ...         return 10
#             ...     @ln.data.Dataset.mosaic_getitem
#             ...     def __getitem__(self, index):
#             ...         return self.enable_mosaic
#         """

#         @wraps(getitem_fn)
#         def wrapper(self, index):
#             if not isinstance(index, int):
#                 self.enable_mosaic = index[0]
#                 index = index[1]

#             ret_val = getitem_fn(self, index)

#             return ret_val

#         return wrapper



# def get_yolox_datadir():
#     """
#     get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
#     this function will return value of the environment variable. Otherwise, use data
#     """
#     yolox_datadir = os.getenv("YOLOX_DATADIR", None)
#     if yolox_datadir is None:
#         import yolox

#         yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
#         yolox_datadir = os.path.join(yolox_path, "datasets")
#     return yolox_datadir

def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            if len(obj["bbox"]) == 0:
                x1,y1,x2,y1 = 0,0,0,0
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
            else:
                x1 = np.max((0, obj["bbox"][0]))
                y1 = np.max((0, obj["bbox"][1]))
                x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
                y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
                if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id



class COCODatasetCvAM(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train_cvam",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        # print("annotation len = {}".format(len(self.annotations)))
        # print("annotation len = {}".format(type(self.annotations)))
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        # l = []
        # for _ids in self.ids:
        #     # print("id = {}".format(_ids))
        #     a = self.load_anno_from_ids(_ids)
        #     l.append(a)
        # return l
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        # print("id = {}, type id = {}".format(id_,type(id_)))
        # if isinstance(id_, str):
        #     id_ = int(id_)
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        # print("annotations = {}".format(annotations[0]))
        patient_id = annotations[0]["patient_id"]
        objs = []
        for obj in annotations:
            if len(obj["bbox"])>0:
                x1 = np.max((0, obj["bbox"][0]))
                y1 = np.max((0, obj["bbox"][1]))
                x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
                y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
                if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    objs.append(obj)
            else:
                x1,y1,x2,y2 = 0,0,0,0
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name, patient_id)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    # def get_img_with_anno_info(self,res, img_info, resized_info, index):
    #     if self.imgs is not None:
    #         pad_img = self.imgs[index]
    #         img = pad_img[: resized_info[0], : resized_info[1], :].copy()
    #     else:
    #         img = self.load_resized_img(index)
    #     return img

    def pull_item_single(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return (img, res.copy(), img_info, np.array([id_]))

    def pull_item(self, index):
        id_ = self.ids[index]
        # print("pull item anno = {}".format(self.annotations[index]))
        _, _, _,_, patient_id = self.annotations[index]
        rest_imgs = {'L_CC':index, 'R_CC':index, 'L_MLO':index, 'R_MLO':index}
        for ind, each in enumerate(self.annotations):
            if each[-1] == patient_id:
                # print("filename = {}".format(each[-2]))
                name_no_ext = each[-2].split('.')[0].split('_')
                img_laterality = name_no_ext[-2]+"_"+name_no_ext[-1]
                # print("filename split = {}".format(img_laterality))
                rest_imgs[img_laterality] = ind
        
        # print("rest_imgs = {}".format(rest_imgs))
        # print("rest_imgs = {}".format(rest_imgs.values()))

        all_return = []
        for each in rest_imgs.values():
            all_return.append(self.pull_item_single(each))


        return all_return

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        # img, target, img_info, img_id = self.pull_item(index)
        all_vars = self.pull_item(index)

        all_return = []
        if self.preproc is not None:
            for (img, target, img_info, img_id) in all_vars:
                img, target = self.preproc(img, target, self.input_dim)
                all_return.append( (img, target, img_info, img_id) )

        return all_return


# if __name__ == '__main__':
#     coco_dataset = COCODatasetCvAM(
#           data_dir=None,
#         json_file="train_cvam.json",
#         name="train_cvam",
#         preproc=None,
#         cache=False,
#     )
#     # ind = coco_dataset.ids[0]
#     for ind in range(len(coco_dataset.ids)):
#         img,target, img_info, img_id = coco_dataset.__getitem__(ind)
#     print(img.shape)
    
import os
import time

from PIL import Image
from torch.utils import data
from joblib import Parallel, delayed
import numpy as np
from torchvision import transforms
import torch


def parallel_load(img_dir, img_list, img_size, n_channel=1, resample="bilinear", verbose=0):
    mode = "L" if n_channel == 1 else "RGB"
    if resample == "bilinear":
        resample = Image.BILINEAR
    elif resample == "nearest":
        resample = Image.NEAREST
    else:
        raise Exception
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert(mode).resize(
            (img_size, img_size), resample=resample))(file) for file in img_list)


class BraTSAD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", test_type = "abnormal", context_encoding=False):
        super(BraTSAD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.test_type = test_type
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test

            if self.test_type == "abnormal":

                
                test_abnormal_dir = os.path.join(self.root, "test", "tumor")
                test_mask_dir = os.path.join(self.root, "test", "annotation")

                test_abnormal = os.listdir(test_abnormal_dir)
                test_masks = [e.replace("flair", "seg") for e in test_abnormal]

                test_l = test_abnormal
                t0 = time.time()

                self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)
                
                self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")  # 0/255

                self.labels += len(test_abnormal) * [1]

                self.img_ids += [img_name.split('.')[0] for img_name in test_l]
                print("Loaded {} test abnormal images. {:.3f}s".format(len(test_abnormal), time.time() - t0))

            else:
            
                test_normal_dir = os.path.join(self.root, "test", "normal")
                test_normal = os.listdir(test_normal_dir)
           
                test_l = test_normal
                t0 = time.time()
                self.slices += parallel_load(test_normal_dir, test_normal, img_size)
                self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
                self.labels += len(test_normal) * [0] #+ len(test_abnormal) * [1]


                self.img_ids += [img_name.split('.')[0] for img_name in test_l]
                print("Loaded {} test normal images. {:.3f}s".format(len(test_normal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)
    

class BraTSData(data.Dataset):
    def __init__(self, main_path, transform=None, mode="train", test_type = "abnormal", context_encoding=False):
        super(BraTSData, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.test_type = test_type
        self.root = main_path
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            

        else:  # test

            if self.test_type == "abnormal":

                
                test_abnormal_dir = os.path.join(self.root, "test", "abnormal", "tumor")
                test_mask_dir = os.path.join(self.root, "test", "abnormal", "annotation")

                test_abnormal = os.listdir(test_abnormal_dir)

                test_l = test_abnormal                

                self.labels += len(test_abnormal) * [1]

                self.img_ids += [img_name.split('.')[0] for img_name in test_l]

            else:
            
                test_normal_dir = os.path.join(self.root, "test", "normal")
                test_normal = os.listdir(test_normal_dir)
           
                test_l = test_normal
                self.labels += len(test_normal) * [0] #+ len(test_abnormal) * [1]

                self.img_ids += [img_name.split('.')[0] for img_name in test_l]

    def __getitem__(self, index):

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":

            data_dir = os.path.join(self.root, "train")

            img = torch.load(os.path.join(data_dir, img_id + ".pt"))

            if self.transform is not None:
                    img = self.transform(img)

            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            if self.test_type == "abnormal":
                test_abnormal_dir = os.path.join(self.root, "test", "abnormal", "tumor")
                test_mask_dir = os.path.join(self.root, "test", "abnormal", "annotation")

                img = torch.load(os.path.join(test_abnormal_dir, img_id + ".pt"))

                if self.transform is not None:
                    img = self.transform(img)
                mask_id = img_id.replace("flair", "seg")
                mask = torch.load(os.path.join(test_mask_dir, mask_id + ".pt"))

                return {'img': img, 'label': label, 'name': img_id, 'mask': mask}
            else:

                test_normal_dir = os.path.join(self.root, "test", "normal")
                img = torch.load(os.path.join(test_normal_dir, img_id + ".pt"))
                if self.transform is not None:
                    img = self.transform(img)
                return {'img': img, 'label': label, 'name': img_id}
           

    def __len__(self):
        return len(self.slices)
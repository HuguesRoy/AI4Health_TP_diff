import os
from torch.utils import data
from torchvision import transforms
import torch


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
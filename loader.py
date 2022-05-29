import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modular import *

class CustomDataset(Dataset):
    def __init__(self, img_list, meta_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_list
        self.meta_list = meta_list
        self.label_list = label_list

    def __getitem__(self, index):
        image = self.img_path_list[index]
        meta = self.meta_list[index]

        # Get image data
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            meta = self.meta_list[index]
            label = self.label_list[index]
            return image, meta, label

        else:
            return image, meta
    
    def __len__(self):
        return len(self.img_path_list)

import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class myDataSet(Dataset):
    def __init__(self, root_dir, type="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            type (string): train data or test data
        """
        self.root_dir = os.path.join(root_dir, "new_%s_set" % type)
        self.transform = transform
        self.type = type

    def __len__(self):
        #the number is according to the image number in the given dataset
        if self.type=="train":
            return 5 
        else:
            return 5

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #load image
        img_name = os.path.join(os.path.join(self.root_dir, "%s_img" % self.type), "%d.tif" % idx)
        image = Image.open(img_name)
        label_name = os.path.join(os.path.join(self.root_dir, "%s_label" % self.type), "%d.png" % idx)
        label = Image.open(label_name)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label
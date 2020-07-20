import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class CustomDataLoader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.names.iloc[index, 0])
        image = io.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image
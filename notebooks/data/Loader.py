import torchvision.transforms as transforms
import data.ImageDataset as Im

from pathlib import Path

def get_data():

    TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    ])
    
    FILE_PATH = '/home/student/Documents/FLORA/notebooks/data/data_set/train'
    TRAIN_TEXT = "leftImg8bit.png"
    GT_TEXT = "labelIds.png"

    entries = Path(FILE_PATH)
        
    X = []
    y = []

    for entry in entries.iterdir():
        if TRAIN_TEXT in str(entry):
            X.append(str(entry))
        elif GT_TEXT in str(entry):
            y.append(str(entry))

    X = sorted(X)
    y = sorted(y)

    data_set = Im.ImageDataset(X=X, y=y, transform=TRANSFORM_IMG)
    return data_set

get_data()
        
    
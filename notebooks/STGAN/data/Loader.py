import torchvision.transforms as transforms
import STGAN.data.DataSet as D

from pathlib import Path


def get_data():
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    FILE_PATH = '/home/student/Documents/FLORA/notebooks/STGAN/data/data_set/'
    TRAIN_TEXT = 'camera'
    GT_TEXT = 'topview'

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

    print(X, y)

    data_set = D.DataSet(X=X, y=y, transform=TRANSFORM_IMG)
    return data_set

get_data()



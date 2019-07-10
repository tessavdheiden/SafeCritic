
import data.Loader as ld
import torch
from torch.utils.data import DataLoader

def test_type():
    train_data = ld.get_data()
    train_data_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=1)
    for (input_img, target_img) in train_data_loader:
        assert type(input_img) == torch.Tensor
        assert type(input_img) == torch.Tensor

def test_size():
    train_data = ld.get_data()
    train_data_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=1)
    assert train_data_loader.batch_size == 5
    for (input_img, target_img) in train_data_loader:
        assert input_img.size() == (5, 3,224,224)
        assert target_img.size() == (5,224,224)

def test_values_output():
    train_data = ld.get_data()
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)

    for (input_img, target_img) in train_data_loader:
        assert (target_img >= torch.zeros_like(target_img)).all()
        assert (target_img <= torch.zeros_like(target_img)+255).all()


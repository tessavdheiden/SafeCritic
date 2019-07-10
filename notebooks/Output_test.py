import model.DeconvNet as md
import torch

def test_load_model():
    net = md.DeconvNet()
    PATH = "/home/student/Documents/FLORA/notebooks/state_dict_test.pth"
    net.load_state_dict(torch.load(PATH))
    for name, param in net.named_parameters():
        print(name, param.sum())
        assert param.sum() > 0 or param.sum() < 0

def test_output():
    net = md.DeconvNet()
    PATH = "/home/student/Documents/FLORA/notebooks/state_dict_test.pth"
    net.load_state_dict(torch.load(PATH))
    zero_input = torch.zeros(224,224)
    output = net(zero_input)
    assert output == zero_input
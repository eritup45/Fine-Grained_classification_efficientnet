# from torchvision.models import efficientnet_b3
import torchvision.models as models
import torch.nn as nn 
from efficientnet_pytorch import EfficientNet
import torch


# TODO: FIX Bug TypeError: forward() takes 1 positional argument but 2 were given
class custom_efficientnet_b3(nn.Module):
    def __init__(self, num_class):
        super(custom_efficientnet_b3, self).__init__()
        eff_model = EfficientNet.from_name('efficientnet-b3')
        eff_model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b3-5fb5a3c3.pth'))
        
        self.base_model = nn.Sequential(*list(eff_model.children())[:-2])
        self.class_model = nn.Linear(eff_model._fc.in_features, num_class)
        self.MemoryEfficientSwish = list(eff_model.children())[-1]
        
    def forward(self, x):
        x = self.base_model(x)
        # x = self.class_model(x)
        # x = self.MemoryEfficientSwish(x)
        return x

# def efficientnet_b3(num_class):
#     model = custom_efficientnet_b3(num_class)
#     return model

"""
# TODO: FIX Bug TypeError: forward() takes 1 positional argument but 2 were given
# Easy Way
def efficientnet_b3(num_class):
    eff_model = EfficientNet.from_name('efficientnet-b3')
    eff_model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b3-5fb5a3c3.pth'))
    
    base_model = nn.Sequential(*list(eff_model.children())[:-2])
    class_model = nn.Linear(eff_model._fc.in_features, num_class)
    MemoryEfficientSwish = list(eff_model.children())[-1]
    model = nn.Sequential(
        base_model,
        class_model,
        MemoryEfficientSwish
    )

    # model = freeze_except_last_layers(model, 140)
    return model
"""

def efficientnet_b1(num_class):
    model = EfficientNet.from_name('efficientnet-b1')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b1-f1951068.pth'))
    model._fc = nn.Linear(model._fc.in_features, num_class)
    # model = freeze_except_last_layers(model, 450)
    return model

def efficientnet_b2(num_class):
    model = EfficientNet.from_name('efficientnet-b2')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b2-8bb594d6.pth'))
    model._fc = nn.Linear(model._fc.in_features, num_class)
    # model = freeze_except_last_layers(model, 450)
    return model

def efficientnet_b3(num_class):
    model = EfficientNet.from_name('efficientnet-b5')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b3-5fb5a3c3.pth'))
    model._fc = nn.Linear(model._fc.in_features, num_class)
    # model = freeze_except_last_layers(model, 450)
    return model

def efficientnet_b4(num_class):
    model = EfficientNet.from_name('efficientnet-b4')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b4-6ed6700e.pth'))
    model._fc = nn.Linear(model._fc.in_features, num_class)
    # model = freeze_except_last_layers(model, 450)
    return model


def efficientnet_b5(num_class):
    model = EfficientNet.from_name('efficientnet-b5')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b5-b6417697.pth'))
    model._fc = nn.Linear(model._fc.in_features, num_class)
    return model

def efficientnet_b7(num_class):
    model = EfficientNet.from_name('efficientnet-b7')
    model.load_state_dict(torch.load('/home/eritup45/NYCU/VRDL/HW/hw1_classification/SOTA/My/model/weights/efficientnet-b7-dcc49843.pth'))
    model._fc.out_features = 200
    model = freeze_except_last_layers(model, 560)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_except_last_layers(model, n):
    # Freeze all layers except the last n layers
    cnt = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        cnt += 1
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= cnt - n:
            param.requires_grad = True
        print(name, param.requires_grad) 
    print(cnt)
    return model

if __name__ == "__main__":
    efficientnet_b3 = efficientnet_b3(200)
    print(efficientnet_b3)

import torch
import torch.nn as nn 
from efficientnet_pytorch import EfficientNet

# 8/6 24:00
# def efficientnet_lite():
#     model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_lite0', pretrained=True)
#     model.classifier.out_features = 2
#     model = freeze_last_layers(model, 140)
#     # model = nn.Sequential(model, nn.Sigmoid())
#     return model

# 8/7 24:00
class probNet(nn.Module):

    def __init__(self):
        super(probNet, self).__init__()
        eff_lite = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_lite0', pretrained=True)
        self.model = nn.Sequential(*list(eff_lite.children())[:-1])
        self.prob_model = nn.Sequential(
            nn.Linear(eff_lite.classifier.in_features, 128),
            nn.Linear(128, 8),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.prob_model(x)  # x.size() = [batch, 1]
        return x


def efficientnet_lite():
    model = probNet()
    return model

def efficientnet_b1():
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc.out_features = 2
    model = freeze_last_layers(model, 140)
    model = nn.Sequential(model, nn.Sigmoid())
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_last_layers(model, n):
    # Freeze the last n layers
    cnt = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        cnt += 1
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= cnt - n:
            param.requires_grad = True
        # print(name, param.requires_grad) 
    return model

if __name__ == '__main__':
    model = efficientnet_lite()
    print(model)
    print('count_parameters: ', count_parameters(model))

    for i,(name, param) in enumerate(model.named_parameters()):
        print(i, name, param.requires_grad) 
        

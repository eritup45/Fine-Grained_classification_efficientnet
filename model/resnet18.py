from torchvision.models import resnet18, resnext50_32x4d
from torch import nn

def resnet_18(num_class):
    # 18 層的 ResNet
#     model = torch.hub.load('pytorch/vision:v0.5.0',
#                            'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, num_class)  # by Fan (Not necessary)

    # Another method
    model = resnet18(pretrained=True)
    model = resnext50_32x4d(pretrained=True)
    # model.fc = nn.Linear(512, num_class)
    model.fc = nn.Linear(2048, num_class)
    # print(model)

    # Load之前訓練model (Optional)
#     if args.pretrained_model_path is not None:
        # model.load_state_dict(torch.load(PRETRAINED_MODEL))
#         model = torch.load(PRETRAINED_MODEL)
    
    return model

def freeze_last_layers(model, n):
    # Freeze the last n layers
    cnt = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        cnt += 1
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= cnt - n:
            param.requires_grad = True
        print(name, param.requires_grad) 
    return model

if __name__ == '__main__':
    freeze_last_layers(resnet_18(200), 0)



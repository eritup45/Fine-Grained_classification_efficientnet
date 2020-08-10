from torchvision.models import resnet18
from torch import nn

def resnet_18(num_class):
    # 18 層的 ResNet
#     model = torch.hub.load('pytorch/vision:v0.5.0',
#                            'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, num_class)  # by Fan (Not necessary)

    # Another method
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)

    # Load之前訓練model (Optional)
#     if args.pretrained_model_path is not None:
        # model.load_state_dict(torch.load(PRETRAINED_MODEL))
#         model = torch.load(PRETRAINED_MODEL)
    
    return model





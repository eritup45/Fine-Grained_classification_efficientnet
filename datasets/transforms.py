from torchvision import transforms
from config      import cfg

def build_train_transform(cfg):
    deg = cfg.DATA.DEGREES
    shift = cfg.DATA.TRANSLATE
    transform = transforms.Compose([
                                    transforms.Resize(cfg.DATA.RESIZE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(degrees=deg, translate=((shift, shift))), # rotate and shift
                                    transforms.ToTensor(),
    ])
    return transform

def build_test_transform(cfg):
    transform = transforms.Compose([
                                    transforms.Resize(cfg.DATA.RESIZE),
                                    transforms.ToTensor(),
    ])
    return transform
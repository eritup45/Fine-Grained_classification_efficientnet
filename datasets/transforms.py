from torchvision import transforms
from config      import cfg
import PIL

def build_train_transform(cfg):
    deg = cfg.DATA.DEGREES
    shift = cfg.DATA.TRANSLATE
    transform = transforms.Compose([
                                    transforms.Resize(cfg.DATA.RESIZE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(
                                        degrees=deg, 
                                        translate=((shift, shift)), 
                                        resample=PIL.Image.BILINEAR 
                                    ), # rotate and shift
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def build_test_transform(cfg):
    transform = transforms.Compose([
                                    transforms.Resize(cfg.DATA.RESIZE, PIL.Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform



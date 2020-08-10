from model      import cnn_model
from model.resnet18 import resnet_18
from model.efficientnet_lite import efficientnet_b1
from model.efficientnet_lite import efficientnet_lite
from config     import cfg
from datasets   import make_test_loader
import torch, os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
# from utils import plot_confusion_matrix
from torch.utils.data         import DataLoader
from PIL import *
import os
from pathlib import Path
from torchvision import transforms
from torch.autograd import Variable

imsize = 528
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU    

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def one_class_test(p, classes):
    model = efficientnet_lite()
    model.load_state_dict(torch.load(weight_path))
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        model.cuda()

    test_folder = Path(p)
    files = list(test_folder.glob('**/*'))
    pred_list = []
    id_list = []
    # files = ['1089.png', '1161.png', '1204.png', '1250.png', '1253.png', '1272.png', '1317.png', '2003.png', '218.png', '2389.png']
    # for i, f in enumerate(files):
    #     files[i] = r'./machine_data/images/test/ng/' + f
    # check_list = []

    with torch.no_grad():
        for f in tqdm(files):
            image = image_loader(str(f))
            pred = model(image)
            pred = pred.to('cpu')
            pred_list.append(pred.max(1)[1])
            id_list.append(str(f))

            # check_list.append(pred)

            torch.cuda.empty_cache()

    pred_list = np.array(pred_list)
    id_list = np.array(id_list)
    
    res = np.hstack((id_list.reshape(-1, 1), pred_list.reshape(-1, 1)))
    np.save(f'{classes}_pred.npy', np.array(res))
    print(res)
    # print(check_list)
    # for i, c in enumerate(check_list):
    #     print(id_list[i], c[0][0], c[0][1])
        
def main():
    one_class_test(r'./machine_data/images/small', 'small')

if __name__ == "__main__":
    main()
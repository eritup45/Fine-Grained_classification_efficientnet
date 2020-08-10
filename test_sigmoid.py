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

model = efficientnet_lite()
# model = resnet_18(num_class=2)
# model = cnn_model()
threshold   = cfg.MODEL.TEST_THRESHOLD
weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

weight = torch.load(weight_path)
model.load_state_dict(weight)

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()

test_loader = make_test_loader(cfg)

model.eval()

test_loss = 0.
correct = 0

# Confused Matrix
if use_cuda:
    all_output = torch.tensor([], dtype=torch.float).cuda()
    all_target = torch.tensor([], dtype=torch.float).cuda()
with torch.no_grad():
    # data = batch data
    # target = batch target
    for data, target in tqdm(test_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output = output.squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy(output.float(), target.float())

        test_loss += loss.item() * data.size(0)
        for i, out in enumerate(output):
            if output[i] < threshold:
                output[i] = torch.tensor(0)
            else:
                output[i] = torch.tensor(1)

        correct += (output == target).sum()
        
        all_target = torch.cat((all_target, target.float()), dim=0)
        all_output = torch.cat((all_output, output), dim=0)
    
    C = confusion_matrix(all_target.cpu().numpy(), all_output.cpu().detach().numpy())
    print(C)
        
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({}/{})'.format(test_loss, accuracy, correct, len(test_loader.dataset)))

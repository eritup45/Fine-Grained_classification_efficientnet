from model      import cnn_model
from model.efficientnet import efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5
from model.resnet18 import resnet_18
from config     import cfg
import torch, os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from datasets   import make_test_loader, find_images_and_labels


# model = cnn_model()
# model = efficientnet_lite()
# model = resnet_18(num_class=cfg.DATA.NUM_CLASS)
model = efficientnet_b2(num_class=cfg.DATA.NUM_CLASS)

weight_path = cfg.MODEL.FINAL_MODEL_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

# submission
submission = []
with open(cfg.PATH.TEST_ANNOTATION) as f:
    test_images = [x.strip() for x in f.readlines()]  # all the testing images

weight = torch.load(weight_path)
model.load_state_dict(weight)

device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
if use_cuda:
    model = model.to(device)
    print(f'Using {device}')

test_loader = make_test_loader(cfg)

model.eval()

# test_loss = 0.
# correct = 0
idx = 0

# Confused Matrix
if use_cuda:
    all_output = torch.tensor([], dtype=torch.float).cuda()
    all_target = torch.tensor([], dtype=torch.float).cuda()
    
test_path   = cfg.PATH.TEST_SET
test_annotation = cfg.PATH.TEST_ANNOTATION
imgs, labels, classes, class_to_idx = find_images_and_labels(test_annotation, test_path)


with torch.no_grad():
    # data = batch data
    # target = batch target
    for data, target in tqdm(test_loader):
        if use_cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        ans_idx = output.max(1)[1].item()
        # Dictionary get key from value
        ans = list(class_to_idx.keys())[list(class_to_idx.values()).index(ans_idx)] 
        submission.append([test_images[idx], ans])
        idx += 1

        # loss = torch.nn.functional.cross_entropy(output, target)
        # test_loss += loss.item() * data.size(0)
        # correct += (output.max(1)[1] == target).sum()
        
        all_target = torch.cat((all_target, target.float()), dim=0)
        all_output = torch.cat((all_output, output), dim=0)
    
    C = confusion_matrix(all_target.cpu().numpy(), all_output.max(1)[1].cpu().detach().numpy())
    print(C)
        
    # test_loss /= len(test_loader.dataset)
    # accuracy = 100. * correct / len(test_loader.dataset)

    # print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({}/{})'.format(test_loss, accuracy, correct, len(test_loader.dataset)))

np.savetxt('answer.txt', submission, fmt='%s')

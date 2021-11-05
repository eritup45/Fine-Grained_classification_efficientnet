from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
# from utils.criterion import Smooth_CELoss # TODO
from model      import cnn_model
from model.resnet18 import resnet_18
from model.efficientnet import efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5
from config      import cfg
from datasets    import make_train_loader

import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn 
import torch.nn.functional as F

# import matplotlib.pyplot as plt

# label smoothing
class Smooth_CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, class_num, label_smooth=0.1):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	# softmax + log
            target = F.one_hot(target, self.class_num)	# one-hot
            
            # label smoothing
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()

def main():
    # model = resnet_18(num_class=cfg.DATA.NUM_CLASS)
    # model = efficientnet_b4(num_class=cfg.DATA.NUM_CLASS)
    model = efficientnet_b2(num_class=cfg.DATA.NUM_CLASS)
    # model = torch.load("./weights/effb1/pre/pre.pt")

    resume_path = cfg.MODEL.RESUME_PATH
    valid_size  = cfg.DATA.VALIDATION_PROPORTION
    epochs      = cfg.MODEL.EPOCH
    lr          = cfg.MODEL.LR
    decay_type  = cfg.MODEL.DECAY_TYPE
    output_path = cfg.MODEL.OUTPUT_PATH
    use_cuda    = cfg.DEVICE.CUDA
    gpu_id      = cfg.DEVICE.GPU

    if resume_path:
        model.load_state_dict(torch.load(resume_path))

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if use_cuda:
        model = model.to(device)
        print(f'Using {device}')

    num_train, train_loader, valid_loader = make_train_loader(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = Smooth_CELoss(cfg.DATA.NUM_CLASS, 0.1)


    # t_total = (Number of data / TRAIN_BATCH_SIZE) * EPOCH
    t_total = (num_train / cfg.DATA.TRAIN_BATCH_SIZE) * epochs
    print(f"LR decay_type: {decay_type}. t_total: {t_total}")
    if decay_type == "warmup_cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.MODEL.WARMUP_STEP, t_total=t_total)
    elif decay_type == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.MODEL.WARMUP_STEP, t_total=t_total)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=4
        )

    min_val_loss = 1000000.
    PATIENCE = 10
    tmp_counter = 0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.
        valid_loss = 0.
        train_correct = 0.
        val_correct = 0.
        # plt_train_losses = []

        # data = batch data
        # target = batch target
        for data, target in tqdm(train_loader):
            # target = label_smoothing(target, cfg.DATA.NUM_CLASS, 0.1)
            if use_cuda:
                data, target = data.to(device), target.to(device)
                # data, target = data.to(device), target.to(device, dtype=torch.long)

            optimizer.zero_grad()
            output = model(data)
            # loss = torch.nn.functional.cross_entropy(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_correct += (output.max(1)[1] == target).sum()
            if decay_type == "warmup_cosine" or decay_type == "warmup_linear":
                scheduler.step()
        # pytorch 1.10之后，学习率调整一般在参数更新之后，即lr_scheduler.step()在optimizer.step()之后调用
        if decay_type == "ReduceLROnPlateau":
            scheduler.step(loss)

        train_acc = 100. * float(train_correct / int(np.floor(len(train_loader.dataset) * (1 - valid_size))))
        print(f'train_acc: {train_acc} = 100. * {train_correct} / {int(np.floor(len(train_loader.dataset) * (1 - valid_size)))}')

        # epoch_loss = train_loss / len(train_loader['train'])
        # plt_train_losses.append(epoch_loss)

        model.eval()
        for data, target in valid_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            valid_loss += loss.item() * data.size(0)
            val_correct += (output.max(1)[1] == target).sum()

        train_loss /= int(np.floor(len(train_loader.dataset) * (1 - valid_size)))
        valid_loss /= int(np.floor(len(valid_loader.dataset) * valid_size))
        acc = 100. * float(val_correct / int(np.floor(len(valid_loader.dataset) * valid_size)))
        print(f'valid_acc: {acc} = 100. * {val_correct} / {int(np.floor(len(valid_loader.dataset) * valid_size))}')
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Val_Accuracy: {:.4f}%, LR: {}'
                .format(epoch, train_loss, valid_loss, acc, 
                    optimizer.state_dict()['param_groups'][0]['lr']))

        # Early stopping by acc
        if valid_loss >= min_val_loss and tmp_counter < PATIENCE:
            tmp_counter += 1
        if valid_loss >= min_val_loss and tmp_counter == PATIENCE:
            print('Early Stopping! Smallest valid loss: {:.4f}, Epoch: {}, LR:{}'
                .format(min_val_loss, epoch-PATIENCE, optimizer.state_dict()['param_groups'][0]['lr']))
            break
        # elif accuracy > best_valid_acc:
        elif valid_loss < min_val_loss:
            min_val_loss = valid_loss
            torch.save(model.state_dict(), 
                f'{output_path}effb2_R5_valin_ep{epoch}_vloss{valid_loss:.3f}_LR{optimizer.state_dict()["param_groups"][0]["lr"]}.pth')
            tmp_counter = 0

    # plt.savefig(plt_train_losses)

if __name__ == '__main__':
    main()

from model      import cnn_model
from model.resnet18 import resnet_18
from model.efficientnet_lite import efficientnet_b1
from model.efficientnet_lite import efficientnet_lite
from config      import cfg
from datasets    import make_train_loader

import torch, os
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt

# model = efficientnet_b1()
model = efficientnet_lite()

resume_path = cfg.MODEL.RESUME_PATH
valid_size  = cfg.DATA.VALIDATION_SIZE
epochs      = cfg.MODEL.EPOCH
lr          = cfg.MODEL.LR
output_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

if resume_path:
    model.load_state_dict(torch.load(resume_path))

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model = model.cuda()

train_loader, valid_loader = make_train_loader(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5
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
    # plt_losses = []

    # data = batch data
    # target = batch target
    for data, target in tqdm(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        train_correct += (output.max(1)[1] == target).sum()

    scheduler.step(loss)

    train_acc = 100. * float(val_correct / int(np.floor(len(train_loader.dataset) * (1 - valid_size))))
    print(f'train_acc: {train_acc} = 100. * {train_correct} / {int(np.floor(len(train_loader.dataset) * (1 - valid_size)))}')

    # epoch_loss = train_loss / len(train_loader['train'])
    # plt_losses.append(epoch_loss)

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
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Val_Accuracy: {:.4f}%'
            .format(epoch, train_loss, valid_loss, acc))

    # Early stopping by acc
    if valid_loss >= min_val_loss and tmp_counter < PATIENCE:
        tmp_counter += 1
    if valid_loss >= min_val_loss and tmp_counter == PATIENCE:
        print('Early Stopping! Smallest valid loss: {:.4f}, Epoch: {}, LR:{}'
              .format(min_val_loss, epoch-PATIENCE, optimizer.state_dict()['param_groups'][0]['lr']))
        break
    # elif accuracy > best_valid_acc:
    elif valid_loss < min_val_loss:
        torch.save(model.state_dict(), output_path)
        min_val_loss = valid_loss
        tmp_counter = 0

# plt.savefig(plt_losses)



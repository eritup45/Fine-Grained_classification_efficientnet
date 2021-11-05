# from .transforms            import build_train_transform, build_test_transform
from .rand_transforms         import build_train_transform, build_test_transform
from torchvision              import datasets
from torch.utils.data         import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

import numpy as np
import os
from PIL import Image

np.random.seed(11)

def find_images_and_labels(annotations_file, img_dir, classes_file = './data/classes.txt', is_train=False):
    """
        Description: 
        Find images and labels from the following data.
        Labels file specify which data is training (testing) data.
        And A file specify how many class is in the data.

        Args: 
        annotations_file: Labels of training (testing) data. (Ex. './data/training_labels.txt')
        img_dir: Images stored in. (Ex. './data/train')

        Return: 
        images_and_labels: (images, labels)
        class_to_idx.keys(): names of class
        class_to_idx: dict() of classes
    """
    img_txt_file = open(annotations_file)
    classes = open(classes_file)
    class_to_idx = dict()
    img_path_list = []
    label_list = []
    labels = []

    # Convert classes to idx
    for idx, c in enumerate(classes):
        class_to_idx[c.split()[0]] = idx 

    if is_train:
        for line in img_txt_file:
            # images path
            filename = line[:].split(' ')[0]
            path = os.path.join(img_dir, filename)
            img_path_list.append(path)

            # labels (converted to numerical)
            label_list.append(line[:-1].split(' ')[-1])
        labels = [class_to_idx[l] for l in label_list]
    if not is_train: 
        for line in img_txt_file:
            # images path
            filename = line[:-1]
            path = os.path.join(img_dir, filename)
            img_path_list.append(path)

            # labels (Just giving arbitary value)
            labels.append(1)

    return img_path_list, labels, class_to_idx.keys(), class_to_idx

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, 
    classes_file = './data/classes.txt', transform=None, is_train=False):
        imgs, labels, _, _ = find_images_and_labels(annotations_file, img_dir, 
            classes_file=classes_file, is_train=is_train)
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = imgs
        self.labels = labels
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx], self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def make_train_loader(cfg):
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TRAIN_BATCH_SIZE
    valid_proportion  = cfg.DATA.VALIDATION_PROPORTION
    train_path  = cfg.PATH.TRAIN_SET
    train_annotation = cfg.PATH.TRAIN_ANNOTATION
    transforms = build_train_transform(cfg)

    trainset = CustomImageDataset(train_annotation, train_path, transform=transforms, is_train=True)
    
    # dataset were splitted in 80 % /20 % for train data and validation data seperately.
    num_train_valid = len(trainset)
    indices = list(range(num_train_valid))
    num_train = int(np.floor(valid_proportion * num_train_valid))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_train:], indices[:num_train]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # If your validation and train data are splitted already, just do
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
    
    return num_train, train_loader, valid_loader

def make_test_loader(cfg):
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TEST_BATCH_SIZE
    test_path   = cfg.PATH.TEST_SET
    test_annotation = cfg.PATH.TEST_ANNOTATION
    transforms = build_test_transform(cfg)
    testset = CustomImageDataset(test_annotation, test_path, transform=transforms)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    return test_loader


if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    num_workers = 4
    batch_size  = 1
    test_path   = "../data/test"
    train_path   = "../data/train"
    test_annotation = "../data/testing_img_order.txt"
    train_annotation = "../data/training_labels.txt"

    trainset = CustomImageDataset(train_annotation, train_path, classes_file = '../data/classes.txt', is_train=True, transform=transform)
    test_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    # testset = CustomImageDataset(test_annotation, test_path, classes_file = '../data/classes.txt', transform=transform)
    # test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    for data, target in test_loader:
        print('data: ', data)
        print('data.size(0): ', data.size(0))
        print('data.size(1): ', data.size(1))
        print('data.size(2): ', data.size(2))
        print('data.size(3): ', data.size(3))
        print('target: ', target)
        print('--------')
    print('len(trainset): ', len(trainset))

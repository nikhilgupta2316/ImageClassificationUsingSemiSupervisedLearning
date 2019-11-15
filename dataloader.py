import PIL
import numpy as np
import torch
from torchvision import transforms

from cifar10 import CIFAR10


def divide_train_supervised_unsupervised(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


def get_train_indices_for_ssl(CIFAR10_dataset, n_labeled):
    train_labeled_idxs, train_unlabeled_idxs = divide_train_supervised_unsupervised(
        CIFAR10_dataset.train_labels, int(n_labeled / 10)
    )
    print("Total Training Data: %d" % len(CIFAR10_dataset.train_labels))
    print("split into")
    print("Labeled Training Data: %d" % len(train_labeled_idxs))
    print("Unlabeled Training Data: %d" % len(train_unlabeled_idxs))
    return train_labeled_idxs, train_unlabeled_idxs


class DataLoader:

    def __init__(self, args):
        self.args = args

        # Initial Data Transform
        transform_img = [
            transforms.ToTensor(),
        ]

        # Normalisation Transform
        if self.args.pretrained:
            normalise = [transforms.Normalize(
                self.args.imagenet_mean_color, self.args.imagenet_std_color
            )]
        else:
            normalise = [transforms.Normalize(
                self.args.cifar10_mean_color, self.args.cifar10_std_color
            )]

        # Model Specific Transform
        if self.args.model == "alexnet":
            resize = [transforms.Resize((224, 224))]
        else:
            resize = []

        # Any Data Augmentation
        data_augmentation = []

        if self.args.data_aug:
            if self.args.model == "resnet":
                data_augmentation = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=0.0, translate=(0.1,0.1), resample=PIL.Image.NEAREST),
                ]
            else:
                data_augmentation = [
                    transforms.RandomCrop(
                        self.args.random_crop_size, padding=self.args.random_crop_pad
                    ),
                    transforms.RandomHorizontalFlip(),
                ]

        # Combine all transformations
        transform_img_train = transform_img + normalise + resize + data_augmentation
        transform_img_test = transform_img + normalise + resize

        transform_train = transforms.Compose(transform_img_train)
        transform_test = transforms.Compose(transform_img_test)

        # Datasets
        if self.args.eval is False:
            self.train_dataset = CIFAR10(
                self.args.cifar10_dir,
                split="train",
                download=True,
                transform=transform_train,
            )
            self.val_dataset = CIFAR10(
                self.args.cifar10_dir,
                split="val",
                download=True,
                transform=transform_test,
            )
        self.test_dataset = CIFAR10(
            self.args.cifar10_dir, split="test", download=True, transform=transform_test
        )

        # Data Loaders
        if self.args.eval is False:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.args.test_batch_size, shuffle=True
            )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.test_batch_size, shuffle=True
        )

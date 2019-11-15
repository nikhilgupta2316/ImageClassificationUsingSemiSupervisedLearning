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
        transform_img_train = data_augmentation + resize + transform_img + normalise
        transform_img_test = resize + transform_img + normalise

        transform_train = transforms.Compose(transform_img_train)
        transform_test = transforms.Compose(transform_img_test)

        # Datasets and DataLoaders
        if self.args.eval is False:
            if self.args.training_mode=="supervised":
                # ########################## Fully Supervised ################################### #
                self.full_supervised_train_dataset = CIFAR10(
                    self.args.cifar10_dir,
                    split="train",
                    download=True,
                    transform=transform_train,
                )
                if self.args.full_data:
                    self.train_dataset = self.full_supervised_train_dataset
                else:
                # ######################## Partially Supervised ################################# #
                    train_labeled_idxs, train_unlabeled_idxs = get_train_indices_for_ssl(self.full_supervised_train_dataset, self.args.train_data_size)
                    self.train_labeled_indices = train_labeled_idxs

                    self.supervised_train_dataset = CIFAR10(
                        self.args.cifar10_dir,
                        split="train",
                        train_split_supervised_indices=np.array(self.train_labeled_indices),
                        download=True,
                        transform=transform_train,
                    )
                    self.train_dataset = self.supervised_train_dataset

                self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=self.args.batch_size, shuffle=True
                )

            elif self.args.training_mode=="semi-supervised":
                # ########################### Semi Supervised ################################### #
                self.full_supervised_train_dataset = CIFAR10(
                    self.args.cifar10_dir,
                    split="train",
                    download=True,
                    transform=transform_train,
                )
                train_labeled_idxs, train_unlabeled_idxs = get_train_indices_for_ssl(self.full_supervised_train_dataset, self.args.train_data_size)
                self.train_labeled_indices = train_labeled_idxs
                self.train_unlabeled_indices = train_unlabeled_idxs

                self.supervised_train_dataset = CIFAR10(
                    self.args.cifar10_dir,
                    split="train",
                    train_split_supervised_indices=np.array(self.train_labeled_indices),
                    download=True,
                    transform=transform_train,
                )
                self.unsupervised_train_dataset = CIFAR10(
                    self.args.cifar10_dir,
                    split="train",
                    train_split_supervised_indices=np.array(self.train_unlabeled_indices),
                    download=True,
                    transform=transform_train,
                )
                self.supervised_train_loader = torch.utils.data.DataLoader(
                    self.supervised_train_dataset, batch_size=self.args.batch_size, shuffle=True
                )
                self.unsupervised_train_loader = torch.utils.data.DataLoader(
                    self.unsupervised_train_dataset, batch_size=self.args.batch_size, shuffle=True
                )
            # ############################## Val Split ########################################## #
            self.val_dataset = CIFAR10(
                self.args.cifar10_dir,
                split="val",
                download=True,
                transform=transform_test,
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.args.test_batch_size, shuffle=True
            )
        # ################################ Test Split ########################################### #
        self.test_dataset = CIFAR10(
            self.args.cifar10_dir, split="test", download=True, transform=transform_test
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.test_batch_size, shuffle=True
        )

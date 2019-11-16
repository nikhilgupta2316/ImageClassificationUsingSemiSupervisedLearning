import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerNN(nn.Module):
    def __init__(self, im_size, n_classes):
        """ Three Layer Neural Network

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        """
        super(ThreeLayerNN, self).__init__()

        flattened_input_dim = im_size[0] * im_size[1] * im_size[2]
        self.layer1 = nn.Linear(flattened_input_dim, 512)
        self.layer2 = nn.Linear(512, 64)
        self.layer3 = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        flattened_images = images.reshape(images.shape[0], -1)
        x = F.relu(self.layer1(flattened_images))
        x = F.relu(self.layer2(x))
        unnormalised_scores = self.layer3(x)
        logits = self.softmax(unnormalised_scores)
        return logits, unnormalised_scores

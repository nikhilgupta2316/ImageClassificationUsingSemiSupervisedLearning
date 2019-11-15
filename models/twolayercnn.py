import torch.nn as nn
import torch.nn.functional as F


class TwoLayerCNN(nn.Module):
    def __init__(self, im_size, n_classes):
        """ TwoLayerCNN Classifier

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        """
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)        
        flattened_dim = 32 * 32 * 64
        self.layer1 = nn.Linear(flattened_dim, n_classes)
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
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        flattened_images = x.reshape(x.shape[0], -1)
        unnormalised_scores = self.layer1(flattened_images)
        logits = self.softmax(unnormalised_scores)
        return logits, unnormalised_scores

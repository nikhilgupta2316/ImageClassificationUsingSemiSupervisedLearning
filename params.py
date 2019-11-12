import argparse
import torch

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="Image Classification on CIFAR-10")

parser.add_argument('--exp-name', type=str, help='Experiment name')
# Data Loading
parser.add_argument("--cifar10-dir", default="data",
                    help="directory that contains cifar-10-batches-py/ "
                         "(downloaded automatically if necessary)")

# Training
parser.add_argument('--optimiser', choices=["sgd", "adam", "rmsprop"], default="sgd",
                    help='Optimizer used for training the model')
parser.add_argument("--learning-rate", type=float, default=0.1, help='Learning rate for optimiser')
parser.add_argument('--lr-scheduler', action='store_true', default=False,
                    help='Use Learning Rate Scheduler to decay LR')
parser.add_argument(
    "--lr-schedule", default=['150', '250'],
    nargs="+",
    help="The epoch(s) at which the learning rate will drop",
)
parser.add_argument("--lr-decay-factor", type=float, default=0.1, help='Learning rate decay factor (Gamma)')
parser.add_argument("--momentum", type=float, default=0.9, help='Momentum for optimiser')
# TODO: Fix help for weight-decay
parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay hyperparameter')

parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
parser.add_argument('--model',
                    choices=['softmax', 'convnet', 'twolayernn', 'densenet', 'vggnet'],
                    help='which model to train/evaluate')

# Data Augmentation
parser.add_argument('--data-aug', action='store_true', default=False,
                    help='Use Data Augmentation during training')
parser.add_argument('--random-crop-size', type=int, default=32,
                        help="Random Crop the images to this size during data augmentation")
parser.add_argument('--random-crop-pad', type=int, default=4,
                        help="Pad the image before Random Crop during data augmentation")

# Testing
parser.add_argument('--eval', action='store_true', default=False,
                    help='Only evaluate saved model')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='Batch size for testing')

# Logging

parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Use Tensorboard for logging')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed for reproducibility')

# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


parser.add_argument('--log-interval', type=int, default=10,
                    help='number of batches between logging train status')

parser.add_argument('--checkpoint-save-interval', type=int, default=1,
                    help='Interval after which checkpoint to save')


def print_args(args):
    s = ""
    args_dict = vars(args)
    for key, value in sorted(args_dict.items()):
        if args.eval is False:
            print(key, value)
        s+= "%s: %s\n" % (key, value)
    return s


def parse_args():
    args = parser.parse_args()
    args.no_of_classes = 10
    args.image_size = (3, 32, 32)

    # Mean Color of training images
    args.cifar10_mean_color = [0.4914, 0.4822, 0.4465]
    # Std Dev of Color of training images
    args.cifar10_std_color = [0.2023, 0.1994, 0.2010]

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

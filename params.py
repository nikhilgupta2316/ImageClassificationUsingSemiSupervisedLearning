import argparse
import torch

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description="Image Classification on CIFAR-10")

parser.add_argument("--exp-name", type=str, help="Experiment name")
# Data Loading
parser.add_argument(
    "--cifar10-dir",
    default="data",
    help="directory that contains cifar-10-batches-py/ "
    "(downloaded automatically if necessary)",
)
# Model
parser.add_argument(
    "--model",
    choices=["softmax", "convnet", "twolayernn", "densenet", "vggnet", "resnet", "alexnet", "onelayernn", "onelayercnn", "twolayercnn"],
    help="which model to train/evaluate",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Use pretrained model from torchvision.models",
)

# Training
parser.add_argument(
    "--training-mode",
    choices=["supervised", "semi-supervised"],
    default="supervised",
    help="Which training method to use",
)
parser.add_argument(
    "--not-full-data",
    action="store_true",
    default=False,
    help="Which training method to use",
)
parser.add_argument(
    "--train-data-size", type=int, default=4000, help="Amount of data to train on in supervised fashion"
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Batch size for training"
)
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
parser.add_argument(
    "--optimiser",
    choices=["sgd", "adam", "rmsprop"],
    default="sgd",
    help="Optimizer used for training the model",
)
parser.add_argument(
    "--learning-rate", type=float, default=0.1, help="Learning rate for optimiser"
)
parser.add_argument(
    "--momentum", type=float, default=0.9, help="Momentum for optimiser"
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0001, help="L2 Regularisation Penalty"
)
parser.add_argument(
    "--lr-scheduler",
    action="store_true",
    default=False,
    help="Use Learning Rate Scheduler to decay LR",
)
parser.add_argument(
    "--lr-lambda-scheduler",
    action="store_true",
    default=False,
    help="Use Learning Rate Scheduler to decay LR",
)
parser.add_argument(
    "--lr-reducer",
    action="store_true",
    default=False,
    help="Use Learning Rate Reducer vary LR when on a Ridge",
)
parser.add_argument(
    "--lr-schedule",
    default=["150", "250"],
    nargs="+",
    help="The epoch(s) at which the learning rate will drop",
)
parser.add_argument(
    "--lr-decay-factor",
    type=float,
    default=0.1,
    help="Learning rate decay factor (Gamma)",
)

# Data Augmentation
parser.add_argument(
    "--data-aug",
    action="store_true",
    default=False,
    help="Use Data Augmentation during training",
)
parser.add_argument(
    "--random-crop-size",
    type=int,
    default=32,
    help="Random Crop the images to this size during data augmentation",
)
parser.add_argument(
    "--random-crop-pad",
    type=int,
    default=4,
    help="Pad the image before Random Crop during data augmentation",
)
# Testing
parser.add_argument(
    "--eval", action="store_true", default=False, help="Only evaluate saved model"
)
parser.add_argument(
    "--eval-checkpoint", type=str, default="", help="Evalute model at checkpoint path"
)
parser.add_argument(
    "--test-batch-size", type=int, default=512, help="Batch size for testing"
)
# Logging
parser.add_argument(
    "--filelogger",
    action="store_true",
    default=False,
    help="Save all the printed info in a file",
)
parser.add_argument(
    "--tensorboard",
    action="store_true",
    default=False,
    help="Use Tensorboard for logging",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    help="number of batches between logging train status",
)
parser.add_argument(
    "--checkpoint-save-interval",
    type=int,
    default=1,
    help="Interval after which checkpoint to save",
)
# Other configuration
parser.add_argument(
    "--seed", type=int, default=1, help="Random seed for reproducibility"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)


def print_args(args):
    s = ""
    args_dict = vars(args)
    for key, value in sorted(args_dict.items()):
        if args.eval is False:
            print(key, value)
        s += "%s: %s\n" % (key, value)
    return s


def parse_args():
    args = parser.parse_args()
    args.no_of_classes = 10
    args.image_size = (3, 32, 32)

    # ImageNet Mean Color
    args.imagenet_mean_color = [0.485, 0.456, 0.406]
    # ImageNet Std Dev Color
    args.imagenet_std_color = [0.229, 0.224, 0.225]

    # CIFAR10 Mean Color
    args.cifar10_mean_color = [0.4914, 0.4822, 0.4465]
    # CIFAR10 Std Dev Color
    args.cifar10_std_color = [0.2023, 0.1994, 0.2010]

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.full_data = not args.not_full_data

    return args

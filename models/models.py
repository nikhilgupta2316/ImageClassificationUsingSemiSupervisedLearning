from models.softmax import Softmax
from models.resnet import ResNet18
from models.alexnet import AlexNet
from models.vggnet import VGGNet
from models.twolayernn import TwoLayerNN
from models.threelayernn import ThreeLayerNN
from models.onelayercnn import OneLayerCNN
from models.twolayercnn import TwoLayerCNN


def Model(args):

    # TODO: Fix args.pretrained
    if args.model == "softmax":
        model = Softmax(args.image_size, args.no_of_classes)
    elif args.model == "twolayernn":
        model = TwoLayerNN(args.image_size, args.no_of_classes)
    elif args.model == "threelayernn":
        model = ThreeLayerNN(args.image_size, args.no_of_classes)
    elif args.model == "onelayercnn":
        model = OneLayerCNN(args.image_size, args.no_of_classes)
    elif args.model == "twolayercnn":
        model = TwoLayerCNN(args.image_size, args.no_of_classes)
    elif args.model == "vggnet":
        model = VGGNet(args.image_size, args.no_of_classes)
    elif args.model == "alexnet":
        model = AlexNet(args.image_size, args.no_of_classes)
    elif args.model == "resnet":
        model = ResNet18()
        # self.model = models.resnet18(pretrained=True)
    else:
        raise Exception("Unknown model {}".format(args.model))

    return model

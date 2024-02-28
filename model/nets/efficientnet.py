from model.efficientnet_pytorch.model import EfficientNet
import argparse
import torchvision.models as models
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='model architecture (default: resnet18)')
args = parser.parse_args()

class build_efficientnet:
    def __init__(self, is_remix=False):
        self.is_remix = is_remix

    def build(self, num_classes):
        if 'efficientnet' in args.arch:
            model = EfficientNet.from_name(args.arch)
        else:
            model = models.__dict__[args.arch]()
        return model
from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .sam2_adapter_light import SAM2_Adapter_Light
from .unet import UNet
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
# from .PVMNet import PVMNet
from .sam2_adapter_tiny import SAM2_Adapter_T
from .segformer import *
from .mobile_sam_adapter import Mobile_sam_adapter
# from .mobile_sam_adapter_change import Mobile_sam_adapter

model_dict = {
    # 'PVMNet': PVMNet,
    'sam2_adapter_tiny': SAM2_Adapter_T,
    'SegFormerB0': make_SegFormerB0,
    'SegFormerB1': make_SegFormerB1,
    'sam2_adapter_light': SAM2_Adapter_Light,
    'unet': UNet,
    'mobile_sam_adapter': Mobile_sam_adapter,
}

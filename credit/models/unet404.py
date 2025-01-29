import segmentation_models_pytorch as smp
import torch
import logging
import copy
import os
import torch.nn.functional as F

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logger = logging.getLogger(__name__)


supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
}

supported_encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x4d",
    "resnext101_32x8d",
    "resnext101_32x16d",
    "resnext101_32x32d",
    "resnext101_32x48d",
    "dpn68",
    "dpn68b",
    "dpn92",
    "dpn98",
    "dpn107",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "inceptionresnetv2",
    "inceptionv4",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "mobilenet_v2",
    "xception",
    "timm-efficientnet-b0",
    "timm-efficientnet-b1",
    "timm-efficientnet-b2",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b5",
    "timm-efficientnet-b6",
    "timm-efficientnet-b7",
    "timm-efficientnet-b8",
    "timm-efficientnet-l2",
    "timm-tf_efficientnet_lite0",
    "timm-tf_efficientnet_lite1",
    "timm-tf_efficientnet_lite2",
    "timm-tf_efficientnet_lite3",
    "timm-tf_efficientnet_lite4",
    "timm-resnest14d",
    "timm-resnest26d",
    "timm-resnest50d",
    "timm-resnest101e",
    "timm-resnest200e",
    "timm-resnest269e",
    "timm-resnest50d_4s2x40d",
    "timm-resnest50d_1s4x24d",
    "timm-res2net50_26w_4s",
    "timm-res2net101_26w_4s",
    "timm-res2net50_26w_6s",
    "timm-res2net50_26w_8s",
    "timm-res2net50_48w_2s",
    "timm-res2net50_14w_8s",
    "timm-res2next50",
    "timm-regnetx_002",
    "timm-regnetx_004",
    "timm-regnetx_006",
    "timm-regnetx_008",
    "timm-regnetx_016",
    "timm-regnetx_032",
    "timm-regnetx_040",
    "timm-regnetx_064",
    "timm-regnetx_080",
    "timm-regnetx_120",
    "timm-regnetx_160",
    "timm-regnetx_320",
    "timm-regnety_002",
    "timm-regnety_004",
    "timm-regnety_006",
    "timm-regnety_008",
    "timm-regnety_016",
    "timm-regnety_032",
    "timm-regnety_040",
    "timm-regnety_064",
    "timm-regnety_080",
    "timm-regnety_120",
    "timm-regnety_160",
    "timm-regnety_320",
    "timm-skresnet18",
    "timm-skresnet34",
    "timm-skresnext50_32x4d",
    "timm-mobilenetv3_large_075",
    "timm-mobilenetv3_large_100",
    "timm-mobilenetv3_large_minimal_100",
    "timm-mobilenetv3_small_075",
    "timm-mobilenetv3_small_100",
    "timm-mobilenetv3_small_minimal_100",
    "timm-gernet_s",
    "timm-gernet_m",
    "timm-gernet_l",
]


def load_premade_encoder_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}"
        )


class SegmentationModel404(torch.nn.Module):
    def __init__(self, conf):
        super(SegmentationModel404, self).__init__()

        self.variables = conf["data"]["variables"]
        self.frames = conf["model"]["frames"]
        self.static_variables = (
            conf["data"]["static_variables"]
            if "static_variables" in conf["data"]
            else []
        )

        in_channels = len(self.variables) + len(self.static_variables)
        out_channels = len(self.variables)

        if conf["model"]["architecture"]["name"] == "unet":
            conf["model"]["architecture"]["decoder_attention_type"] = "scse"
        conf["model"]["architecture"]["in_channels"] = in_channels
        conf["model"]["architecture"]["classes"] = out_channels

        self.model = load_premade_encoder_model(conf["model"]["architecture"])

    def forward(self, x):
        x = F.avg_pool3d(x, kernel_size=(2, 1, 1)) if x.shape[2] > 1 else x
        x = x.squeeze(2)  # squeeze time dim
        x = self.model(x)
        return x.unsqueeze(2)

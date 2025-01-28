import logging

import torch
from credit.models.base_model import BaseModel
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)

class DebuggerModel(BaseModel):
    def __init__(
        self,
        image_height: int = 640,
        image_width: int = 1280,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        post_conf: dict = None,
        **kwargs,
    ):
        """
        A dummy debugger model for testing other parts of the infrastructure.
        Input to class init is to check if params are in the config
        Inputs require the minimal input args common to all models.
        Has one trainable param (self.coef) in order to make sure trainer does not error out.
        prediction is self.coef * x
        """
        super().__init__()
        
        # check if these params are in the config
        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels
        self.input_only_channels = input_only_channels
        self.output_only_channels = output_only_channels


        # input channels
        input_channels = channels * levels + surface_channels + input_only_channels
        # output channels
        output_channels = channels * levels + surface_channels + output_only_channels

        # trainable weights for "prediction" of coef * x
        # trainer will error out if there are no trainable parameters
        self.linear = torch.nn.Linear(input_channels, output_channels, bias=False)

        if post_conf is None:
            post_conf = {"activate": False}

        self.use_post_block = post_conf['activate']

        if self.use_post_block:
            # freeze base model weights before postblock init
            if (post_conf["skebs"].get("activate", False) 
                and post_conf["skebs"].get("freeze_base_model_weights", False)):
                logger.warning("freezing all base model weights due to skebs config")
                for param in self.parameters():
                    param.requires_grad = False
            
            logger.info("using postblock")
            self.postblock = PostBlock(post_conf)

    def forward(self, x):
        """
            forward that multiplies self.coef to the input
            used to test postblock and other model parts
        """
        x_copy = None
        if self.use_post_block:  # copy tensor to feed into postBlock later
            x_copy = x.clone().detach()

        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.permute(0, -1, 1, 2, 3)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x

if __name__ == "__main__":
    image_height = 640  # 640, 192
    image_width = 1280  # 1280, 288
    levels = 15
    frames = 2
    channels = 4
    surface_channels = 7
    input_only_channels = 3
    frame_patch_size = 2

    input_tensor = torch.randn(
        1,
        channels * levels + surface_channels + input_only_channels,
        frames,
        image_height,
        image_width,
    ).to("cuda")

    model = DebuggerModel(
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        levels=levels,
        post_conf=None
    ).to("cuda")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    y_pred = model(input_tensor.to("cuda"))
    print("Predicted shape:", y_pred.shape)

    # print(model.rk4(input_tensor.to("cpu")).shape)
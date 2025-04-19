from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F


class MSAPredictor:
    def __init__(self, model):
        self.model = model

    def set_image(self, image):
        if len(image.shape) != 3 or image.shape[2] not in [1, 3]:
            raise ValueError(
                "Input image must have shape (H, W, C) where C is 1 or 3.")
        image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension to make BCHW
        self.set_torch_image(torch.from_numpy(image).float().to(device=self.model.device))

    def set_torch_image(self, image, original_image_size=None):
        self.img_feature = None
        self.original_image_size = original_image_size
        self.img_feature = self.model.image_encoder(image)

    def predict_torch(self,
                      point_coords: Optional[torch.Tensor] = None,
                      point_labels: Optional[torch.Tensor] = None,
                      boxes: Optional[torch.Tensor] = None,
                      mask_input: Optional[torch.Tensor] = None,
                      multimask_output: bool = False):
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        se, de = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        pred, _ = self.model.mask_decoder(
            image_embeddings=self.img_feature,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=multimask_output,
        )
        pred = F.interpolate(pred, size=(
            self.model.image_encoder.img_size, self.model.image_encoder.img_size))
        return pred, _, None

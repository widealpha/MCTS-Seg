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
        # Add batch dimension to make BCHW
        image = np.expand_dims(image, axis=0)
        self.set_torch_image(torch.from_numpy(
            image).float().to(device=self.device))

    def set_torch_image(self, image, original_image_size=None):
        self.img_feature = None
        self.original_image_size = original_image_size
        self.img_feature = self.model.image_encoder(image)

    def predict(self,
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = False,
                return_logits: bool = False
                ):
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None,
                                                      :, :], labels_torch[None, :]
        if box is not None:
            box_torch = torch.as_tensor(
                box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    def predict_torch(self,
                      point_coords: Optional[torch.Tensor] = None,
                      point_labels: Optional[torch.Tensor] = None,
                      boxes: Optional[torch.Tensor] = None,
                      mask_input: Optional[torch.Tensor] = None,
                      multimask_output: bool = False,
                      return_logits: bool = False):
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        se, de = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.img_feature,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=multimask_output,
        )
        pred = F.interpolate(low_res_masks, size=(
            self.model.image_encoder.img_size, self.model.image_encoder.img_size))
        return pred, iou_predictions, low_res_masks

    @property
    def device(self) -> torch.device:
        return self.model.device
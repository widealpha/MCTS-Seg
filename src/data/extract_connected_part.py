import os
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm


def extract_largest_connected_component(in_dir, out_dir):
    point_masks_dir = in_dir
    connected_masks_dir = out_dir

    if not os.path.exists(connected_masks_dir):
        os.makedirs(connected_masks_dir)

    mask_files = [f for f in os.listdir(point_masks_dir) if f.endswith('.png')]
    mask_files.sort()
    for mask_file in tqdm(mask_files, desc="Extract largest connected component"):
        mask_path = os.path.join(point_masks_dir, mask_file)
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)

        # Label connected components
        labeled_mask, num_features = label(mask_array, return_num=True)

        # Find the largest connected component
        largest_component = None
        max_area = 0
        for region in regionprops(labeled_mask):
            if region.area > max_area:
                max_area = region.area
                largest_component = region

        if largest_component is not None:
            largest_component_mask = (
                labeled_mask == largest_component.label).astype(np.uint8) * 255

            # Label the holes in the largest component
            inverted_mask = np.logical_not(largest_component_mask // 255)
            labeled_holes, num_holes = label(inverted_mask, return_num=True)

            # Fill holes smaller than 10% of the largest component area
            for region in regionprops(labeled_holes):
                if region.area < max_area * 0.1:
                    coords = region.coords
                    for coord in coords:
                        largest_component_mask[coord[0], coord[1]] = 255

            largest_component_image = Image.fromarray(largest_component_mask)
            largest_component_image.save(
                os.path.join(connected_masks_dir, mask_file))


if __name__ == '__main__':
    extract_largest_connected_component()

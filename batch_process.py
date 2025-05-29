# THIS IS BATCH PROCESSING; SINGLE CORE

import os
from stylizer import stylize_image

input_folder = "input_folder"
output_folder = "output_folder"

os.makedirs(output_folder, exist_ok=True)
prev_fg_mask_np = None
prev_mask = None
prev_gray = None

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        _, _, prev_fg_mask_np, prev_gray = stylize_image(
            input_path, output_path, prev_fg_mask_np, prev_gray
        )

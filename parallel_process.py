import os
from concurrent.futures import ProcessPoolExecutor
from stylizer import stylize_image, load_image, get_foreground_mask
from tqdm import tqdm
import numpy as np
import cv2

input_folder = "input_folder"
output_folder = "output_folder"
video_output_path = "video_output/final_video.mp4"
fps = 24
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

image_paths = [
    (os.path.join(input_folder, filename), os.path.join(output_folder, filename))
    for filename in sorted(os.listdir(input_folder))
    if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
]


def process_image(args):
    input_path, output_path, shared_fg_mask_np = args
    stylize_image(input_path, output_path, shared_fg_mask_np)


def frames_to_video(frames_folder: str, output_path: str, fps: int):
    image_files = sorted([
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise ValueError("No frames found for video generation.")

    first_frame = cv2.imread(os.path.join(frames_folder, image_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in tqdm(image_files, desc="🎥 Creating video"):
        frame = cv2.imread(os.path.join(frames_folder, f))
        writer.write(frame)

    writer.release()
    print(f"✅ Video saved to {output_path}")


if __name__ == "__main__":
    first_input_path = image_paths[0][0]
    first_image = load_image(first_input_path)
    shared_fg_mask_np = np.array(get_foreground_mask(first_image)) // 255

    image_paths_with_mask = [
        (in_path, out_path, shared_fg_mask_np) for in_path, out_path in image_paths
    ]

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, image_paths_with_mask), total=len(image_paths)))

    frames_to_video(output_folder, video_output_path, fps)
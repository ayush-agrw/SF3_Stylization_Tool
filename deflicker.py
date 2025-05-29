import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import VideoClip

input_path = "video_output/final_video.mp4"
output_path = "video_output/final_video_deflickered.mp4"
clip = VideoFileClip(input_path)

# Set your motion threshold (tweak for best result)
MOTION_THRESH = 10


def motion_aware_deflicker(get_frame, t):
    """
    Apply motion-aware deflickering to a frame at time t.
    Works by comparing current frame with neighbors in time.
    """
    fps = clip.fps
    window = 1 / fps
    t_minus = max(t - window, 0)
    t_plus = min(t + window, clip.duration)

    # Grab three frames
    f_minus = get_frame(t_minus).astype(np.int16)
    f_curr = get_frame(t).astype(np.int16)
    f_plus = get_frame(t_plus).astype(np.int16)

    # Compute motion mask
    diff1 = np.abs(f_curr - f_minus)
    diff2 = np.abs(f_curr - f_plus)
    motion_map = np.maximum(diff1, diff2).mean(axis=2)
    mask = (motion_map < MOTION_THRESH).astype(bool)

    # Smoothed candidate
    smooth = ((f_minus + f_curr + f_plus) / 3).astype(np.uint8)

    # Build output: static pixels from smooth, moving from original
    out = f_curr.astype(np.uint8)
    out[mask] = smooth[mask]
    return out


# Create a function that can be applied to the video
def make_frame(t):
    return motion_aware_deflicker(clip.get_frame, t)


# Create a new video with the deflickered frames
deflickered = VideoClip(make_frame, duration=clip.duration)

# Copy audio from original clip if it exists
if clip.audio is not None:
    deflickered = deflickered.set_audio(clip.audio)

# Set fps to match the original clip
deflickered = deflickered.with_fps(clip.fps)

# Write the deflickered video
deflickered.write_videofile(
    output_path,
    codec='libx264',
    audio_codec='aac',
    preset='medium',
    bitrate='3000k'
)
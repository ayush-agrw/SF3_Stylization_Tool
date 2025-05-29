from PIL import Image, ImageDraw, ImageTk, ImageFilter, ImageChops
import numpy as np
import cv2
from rembg import remove
import tkinter as tk
from tkinter import Scale, Label
import os


# --------------------------
# Utility Functions
# --------------------------


def load_image(path):
    return Image.open(path).convert("RGBA")


def save_image(image, path):
    image.save(path)


def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)


def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))


def create_circular_fade_mask(width, height, strength=1.0):
    """
    Create a radial alpha mask fading out toward the edges.
    Strength controls how strongly it fades.
    """
    x = np.linspace(-1, 1, width)[np.newaxis, :]
    y = np.linspace(-1, 1, height)[:, np.newaxis]
    distance = np.sqrt(x ** 2 + y ** 2)

    mask = 1.0 - np.clip((distance - 0.4) * 1.2 * strength, 0, 1)
    mask = (mask * 255).astype(np.uint8)

    return Image.fromarray(mask, mode="L")


def saturation_blend(base: Image.Image, overlay: Image.Image) -> Image.Image:
    base_np = np.array(base.convert("RGBA")).astype(np.float32) / 255.0
    overlay_np = np.array(overlay.convert("RGBA")).astype(np.float32) / 255.0

    base_rgb = base_np[..., :3]
    overlay_rgb = overlay_np[..., :3]
    alpha = overlay_np[..., 3:]

    # Convert to HSV
    base_hsv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2HSV)
    overlay_hsv = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2HSV)

    # Replace saturation
    blended_hsv = np.copy(base_hsv)
    blended_hsv[..., 1] = overlay_hsv[..., 1] * alpha[..., 0] + blended_hsv[..., 1] * (1 - alpha[..., 0])

    # Convert back to RGB
    blended_rgb = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2RGB)
    blended_rgba = np.dstack([blended_rgb, base_np[..., 3]]) * 255
    return Image.fromarray(blended_rgba.astype(np.uint8), mode="RGBA")


def get_foreground_mask(image: Image.Image) -> Image.Image:
    """
    Returns a binary mask (L mode) of the foreground.
    """
    fg_removed = remove(image)
    alpha = fg_removed.split()[-1]  # Alpha channel is the mask
    return alpha.point(lambda p: 255 if p > 20 else 0)  # Threshold to binary


def smooth_mask(mask_np, prev_mask_np=None, alpha=0.5):
    # Apply spatial blur
    blurred = cv2.GaussianBlur(mask_np.astype(np.float32), (5, 5), 0)

    if prev_mask_np is not None:
        # Temporal blend with previous mask
        blurred = alpha * prev_mask_np + (1 - alpha) * blurred

    # Re-binarize
    return (blurred > 0.5).astype(np.uint8)


def warp_mask_with_flow(prev_mask, prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = prev_mask.shape
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.stack(flow_map, axis=-1).astype(np.float32)
    remap = flow_map + flow

    warped = cv2.remap(prev_mask.astype(np.float32), remap[..., 0], remap[..., 1], interpolation=cv2.INTER_LINEAR)
    return warped


class BrushGUI:
    def __init__(self, root, image_path):
        self.root = root
        self.original = load_image(image_path)
        self.region = [710, 390, 1210, 690]  # initial region

        # Main layout: left for sliders, right for image
        self.frame = tk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        self.slider_frame = tk.Frame(self.frame)
        self.slider_frame.pack(side="left", padx=10, pady=10)

        self.label = Label(self.frame)
        self.label.pack(side="right", padx=10, pady=10)

        # Vertical sliders
        self.x0 = Scale(self.slider_frame, from_=0, to=self.original.width, label="x0", orient="vertical", command=self.update_region)
        self.x1 = Scale(self.slider_frame, from_=0, to=self.original.width, label="x1", orient="vertical", command=self.update_region)
        self.y0 = Scale(self.slider_frame, from_=0, to=self.original.height, label="y0", orient="vertical", command=self.update_region)
        self.y1 = Scale(self.slider_frame, from_=0, to=self.original.height, label="y1", orient="vertical", command=self.update_region)

        for s, val in zip([self.x0, self.y0, self.x1, self.y1], self.region):
            s.set(val)
            s.pack(side="left", fill="y", expand=True)

        self.update_preview()

    def update_region(self, _=None):
        self.region = [self.x0.get(), self.y0.get(), self.x1.get(), self.y1.get()]
        self.update_preview()

    def update_preview(self):
        img = self.original.copy()
        stylized = apply_dotted_brush_strokes(img, tuple(self.region))
        preview = stylized.resize((960, int(960 * img.height / img.width)))
        img_tk = ImageTk.PhotoImage(preview)
        self.label.config(image=img_tk)
        self.label.image = img_tk

# --------------------------
# Stylization Layer Functions
# --------------------------

#ADJUST:
# apply_halftone: rows and cols
# apply_dotted_brush_strokes: dot spacing, radius = (dot_spacing) * (1 - brightness)
# apply_gradient_overlay: vignette = 1 - radius ** 10
# stylize_image: stylization layers


def apply_halftone(image: Image.Image) -> Image.Image:
    """
    Apply halftone texture using saturation_blend on 8 equal regions (4 cols × 2 rows)
    with smooth fading on tile edges to avoid harsh borders.
    """
    texture_path = "assets/halftone.png"
    try:
        texture = Image.open(texture_path).convert("RGBA")
    except FileNotFoundError:
        print("❌ Halftone texture not found.")
        return image

    img_w, img_h = image.size
    cols, rows = 2, 2
    part_w, part_h = img_w // cols, img_h // rows

    final_image = Image.new("RGBA", (img_w, img_h))

    for row in range(rows):
        for col in range(cols):
            left = col * part_w
            upper = row * part_h
            right = left + part_w
            lower = upper + part_h

            region = image.crop((left, upper, right, lower))

            # Tile halftone texture without scaling
            tiled_texture = Image.new("RGBA", (part_w, part_h))
            tex_w, tex_h = texture.size
            for x in range(0, part_w, tex_w):
                for y in range(0, part_h, tex_h):
                    tiled_texture.paste(texture, (x, y), texture)

            # Create and apply circular fade mask
            fade_mask = create_circular_fade_mask(part_w, part_h, strength=1.0)
            r, g, b, a = tiled_texture.split()
            a = Image.composite(a, fade_mask, fade_mask)
            tiled_texture = Image.merge("RGBA", (r, g, b, a))

            # Apply blend
            blended = saturation_blend(region, tiled_texture)
            final_image.paste(blended, (left, upper))
    return final_image


def apply_dotted_brush_strokes(image: Image.Image, region_box=(0, 0, 0, 0)) -> Image.Image:
    """
    Overlay dry brush textures (e.g., on characters or shadows).
    """
    # TODO: Load brush texture PNGs and blend with random transforms
    dot_spacing = 10
    img_w, img_h = image.size
    base_np = np.array(image.convert("RGB"))

    # Get foreground mask to exclude brush strokes on subject
    fg_mask = get_foreground_mask(image)

    # Create empty transparent layer for the strokes
    stroke_layer = Image.new("RGBA", (img_w, img_h))
    draw = ImageDraw.Draw(stroke_layer)

    x0, y0, x1, y1 = region_box

    for y in range(y0, y1, dot_spacing):
        for x in range(x0, x1, dot_spacing):
            if fg_mask.getpixel((x, y)) > 0:
                continue  # Skip foreground

            px0 = max(x - dot_spacing // 2, 0)
            py0 = max(y - dot_spacing // 2, 0)
            px1 = min(x + dot_spacing // 2, img_w)
            py1 = min(y + dot_spacing // 2, img_h)
            patch = base_np[py0:py1, px0:px1]

            if patch.size == 0:
                continue

            avg_color = patch.mean(axis=(0, 1))
            brightness = np.mean(avg_color) / 255.0
            radius = (dot_spacing) * (1 - brightness)

            if radius > 1:
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=tuple(avg_color.astype(int)) + (255,)
                )

    # Create a circular fade mask matching the region size
    region_w = x1 - x0
    region_h = y1 - y0
    fade_mask = create_circular_fade_mask(region_w, region_h, strength=1.2)

    # Paste the mask onto a full-size mask at the right region
    full_mask = Image.new("L", (img_w, img_h), 0)
    full_mask.paste(fade_mask, (x0, y0))

    # Apply fade mask to alpha channel of stroke_layer
    r, g, b, a = stroke_layer.split()
    a = Image.composite(a, full_mask, full_mask)
    faded_stroke_layer = Image.merge("RGBA", (r, g, b, a))

    result = Image.alpha_composite(image, faded_stroke_layer)
    return result


def apply_noise_overlay(image: Image.Image, intensity=0.04) -> Image.Image:
    """
    Apply film grain or noise over the entire image.
    :param intensity:
    """
    # TODO: Generate or load noise texture and blend
    img = image.convert("RGBA")
    np_img = np.array(img).astype(np.int16)  # use int16 to avoid overflow

    # Generate noise in range [-intensity*255, +intensity*255]
    noise = np.random.uniform(-intensity * 255, intensity * 255, np_img.shape[:2] + (3,))
    np_img[..., :3] += noise.astype(np.int16)

    # Clip values and convert back to uint8
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    # Preserve original alpha channel
    result = Image.fromarray(np.dstack((np_img[..., :3], np.array(img)[..., 3])), mode="RGBA")
    return result


def apply_splatter(image: Image.Image) -> Image.Image:
    """
    Apply spray paint splatter textures to edges or dark regions.
    """
    # TODO: Load splatter PNGs, position randomly, blend
    return image


# WORKS WITH PARALLEL PROCESSING! CAN PRODUCE JITTER...
def apply_glitch_effect(image: Image.Image, shared_fg_mask_np=None) -> (Image.Image, np.ndarray, np.ndarray):
    """
    Apply chromatic aberration glitch effect with optional fixed foreground mask.
    """
    img_w, img_h = image.size
    image_np = np.array(image.convert("RGBA"))

    # Step 1: Get foreground mask
    if shared_fg_mask_np is None:
        fg_mask_img = get_foreground_mask(image)  # L mode (0 or 255)
        fg_mask_np = np.array(fg_mask_img) // 255
    else:
        fg_mask_np = shared_fg_mask_np

    # Step 2: Background mask
    bg_mask_np = 1 - fg_mask_np

    # Step 3: Detect bright areas (on full image)
    brightness = image_np[..., :3].mean(axis=2)
    light_mask = (brightness > 100).astype(np.uint8)

    # Combine with background mask to get glitch area
    glitch_mask = (bg_mask_np * light_mask).astype(np.uint8)

    # Step 4: Split channels
    r, g, b, a = cv2.split(image_np)

    def shift_channel(channel, dx, dy):
        shifted = np.roll(channel, dx, axis=1)
        shifted = np.roll(shifted, dy, axis=0)
        return np.where(glitch_mask == 1, shifted, channel)

    r_glitched = shift_channel(r, dx=4, dy=0)
    g_glitched = shift_channel(g, dx=0, dy=4)
    b_glitched = shift_channel(b, dx=0, dy=0)

    glitched_np = cv2.merge([r_glitched, g_glitched, b_glitched, a])
    glitched_img = Image.fromarray(glitched_np, mode="RGBA")

    return glitched_img, glitched_np, fg_mask_np


# USE BATCH PROCESSING FOR STATIC MASK!
def apply_glitch_effect_dynamic(image: Image.Image, prev_fg_mask_np=None, prev_gray=None):
    image_np = np.array(image.convert("RGBA"))
    image_gray = cv2.cvtColor(image_np[..., :3], cv2.COLOR_RGB2GRAY)

    fg_mask_img = get_foreground_mask(image)
    fg_mask_np = np.array(fg_mask_img) // 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    fg_mask_np = cv2.morphologyEx(fg_mask_np, cv2.MORPH_OPEN, kernel)
    fg_mask_np = cv2.morphologyEx(fg_mask_np, cv2.MORPH_CLOSE, kernel)

    # Warp previous mask using optical flow
    if prev_fg_mask_np is not None and prev_gray is not None:
        warped_prev_mask = warp_mask_with_flow(prev_fg_mask_np, prev_gray, image_gray)
        fg_mask_np = (0.6 * warped_prev_mask + 0.4 * fg_mask_np).astype(np.float32)

    # Smooth & threshold
    fg_mask_np = cv2.GaussianBlur(fg_mask_np, (11, 11), 0)
    fg_mask_np = (fg_mask_np > 0.5).astype(np.uint8)
    bg_mask_np = 1 - fg_mask_np

    brightness = image_np[..., :3].mean(axis=2)
    light_mask = (brightness > 100).astype(np.uint8)
    glitch_mask = (bg_mask_np * light_mask).astype(np.uint8)

    r, g, b, a = cv2.split(image_np)

    def shift_channel(channel, dx, dy):
        shifted = np.roll(channel, dx, axis=1)
        shifted = np.roll(shifted, dy, axis=0)
        return np.where(glitch_mask == 1, shifted, channel)

    r_glitched = shift_channel(r, dx=4, dy=0)
    g_glitched = shift_channel(g, dx=0, dy=4)
    b_glitched = shift_channel(b, dx=0, dy=0)

    glitched_np = cv2.merge([r_glitched, g_glitched, b_glitched, a])
    glitched_img = Image.fromarray(glitched_np, mode="RGBA")

    return glitched_img, glitched_np, fg_mask_np, image_gray


def apply_gradient_overlay(image: Image.Image) -> Image.Image:
    """
    Overlay a mood gradient over the image with corrected aspect ratio handling.
    """
    width, height = image.size
    aspect_ratio = width / height

    # Create meshgrid with aspect ratio compensation
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Correct for non-square images
    xx *= aspect_ratio  # Stretch x to match scale of y

    # Calculate radial distance and normalize
    radius = np.sqrt(xx ** 2 + yy ** 2)
    radius = radius / np.max(radius)

    # Vignette strength
    vignette = 1 - radius ** 8
    vignette = np.clip(vignette, 0, 1)
    vignette = (vignette * 255).astype(np.uint8)

    # Convert to mask and blur
    vignette_mask = Image.fromarray(vignette).filter(ImageFilter.GaussianBlur(radius=width // 8))

    # Overlay
    black_overlay = Image.new("RGB", (width, height), (0, 0, 0))
    final = Image.composite(image, black_overlay, vignette_mask)

    return final


def apply_bokeh(image: Image.Image) -> Image.Image:
    """
    Add bokeh light overlays to light sources or edges.
    """
    # TODO: Load bokeh PNGs, apply to top layer with light blending
    return image


# --------------------------
# Stylization Pipeline
# --------------------------


# def stylize_image(input_path, output_path, prev_fg_mask_np=None, prev_gray=None):
def stylize_image(input_path, output_path, shared_fg_mask_np=None):
    image = load_image(input_path)

    image, glitched_np, fg_mask_np = apply_glitch_effect(image, shared_fg_mask_np)
    # image, glitched_np, new_fg_mask_np, new_gray = apply_glitch_effect_dynamic(
    #     image, prev_fg_mask_np, prev_gray
    # )

    # image = apply_dotted_brush_strokes(image, region_box=(0, 540, 1920, 1080))  # bottom half
    # image = apply_dotted_brush_strokes(image, region_box=(310, 139, 1610, 942))  # middle part
    # image = apply_dotted_brush_strokes(image, region_box=(0, 0, image.width, image.height))  # full
    # image = apply_dotted_brush_strokes(image, region_box=(710, 390, 1210, 690))  # smaller middle part

    # print("Type of image before halftone:", type(image))
    image = apply_halftone(image)
    image = apply_gradient_overlay(image)
    image = apply_noise_overlay(image)

    # image = apply_splatter(image)
    # image = apply_bokeh(image)

    save_image(image, output_path)
    print(f"✅ Saved stylized image to {output_path}")
    # return image, glitched_np, new_fg_mask_np, new_gray
    return image, glitched_np, fg_mask_np


# --------------------------
# Entry Point
# --------------------------


if __name__ == "__main__":
    #root = tk.Tk()
    #root.title = ("Region")
    #app = BrushGUI(root, "input6.webp")  # change image path if needed
    #root.mainloop()
    #input_img = "input6.webp"
    input_img = "input2.webp"
    output_img = "output.png"
    stylize_image(input_img, output_img)

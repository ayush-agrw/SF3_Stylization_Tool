from PIL import Image, ImageDraw
import numpy as np
import cv2

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

# --------------------------
# Stylization Layer Functions
# --------------------------

def apply_halftone(image: Image.Image, dot_spacing=25) -> Image.Image:
    """
    Automatically apply colored halftone based on brightness and local colors.
    """
    img_w, img_h = image.size
    base_np = np.array(image.convert("RGB"))

    # Create an empty transparent image
    halftone_img = Image.new("RGBA", (img_w, img_h))
    draw = ImageDraw.Draw(halftone_img)

    for y in range(0, img_h, dot_spacing):
        for x in range(0, img_w, dot_spacing):
            # Get the average color in a small neighborhood
            x0 = max(x - dot_spacing // 2, 0)
            y0 = max(y - dot_spacing // 2, 0)
            x1 = min(x + dot_spacing // 2, img_w)
            y1 = min(y + dot_spacing // 2, img_h)
            patch = base_np[y0:y1, x0:x1]

            if patch.size == 0:
                continue

            avg_color = patch.mean(axis=(0,1))
            brightness = np.mean(avg_color) / 255.0

            # Dot size based on brightness (darker areas -> bigger dots)
            max_radius = dot_spacing // 2
            radius = max_radius * (1 - brightness)

            if radius > 1:  # Only draw visible dots
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=tuple(avg_color.astype(int)) + (255,)
                )

    # Composite the halftone over the original image
    result = Image.alpha_composite(image, halftone_img)
    return result


def apply_brush_strokes(image: Image.Image) -> Image.Image:
    """
    Overlay dry brush textures (placeholder).
    """
    return image

def apply_noise_overlay(image: Image.Image) -> Image.Image:
    """
    Apply film grain or noise (placeholder).
    """
    return image

def apply_splatter(image: Image.Image) -> Image.Image:
    """
    Apply spray paint splatter (placeholder).
    """
    return image

def apply_glitch_effect(image: Image.Image) -> Image.Image:
    """
    Apply glitch effects (placeholder).
    """
    return image

def apply_gradient_overlay(image: Image.Image) -> Image.Image:
    """
    Apply gradient overlay (placeholder).
    """
    return image

def apply_bokeh(image: Image.Image) -> Image.Image:
    """
    Add bokeh lights (placeholder).
    """
    return image

# --------------------------
# Stylization Pipeline
# --------------------------

def stylize_image(input_path: str, output_path: str):
    image = load_image(input_path)

    image = apply_halftone(image)
    image = apply_brush_strokes(image)
    image = apply_noise_overlay(image)
    image = apply_splatter(image)
    image = apply_glitch_effect(image)
    image = apply_gradient_overlay(image)
    image = apply_bokeh(image)

    save_image(image, output_path)
    print(f"✅ Saved stylized image to {output_path}")

# --------------------------
# Entry Point
# --------------------------

if __name__ == "__main__":
    input_img = "input6.webp"
    output_img = "output.png"
    stylize_image(input_img, output_img)

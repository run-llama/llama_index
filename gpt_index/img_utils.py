# Convert Image to Base64
import base64
from io import BytesIO

from PIL import Image


def img_2_b64(image: Image, format: str = "JPEG") -> str:
    """Convert a PIL.Image to a base64 encoded image str."""
    buff = BytesIO()
    image.save(buff, format=format)
    img_str = base64.b64encode(buff.getvalue())
    return img_str


def b64_2_img(data: str) -> Image:
    """Convert base64 encoded image str to a PIL.Image."""
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

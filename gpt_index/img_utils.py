# Convert Image to Base64
import base64
from io import BytesIO


def im_2_b64(image, format: str = "JPEG"):
    buff = BytesIO()
    image.save(buff, format=format)
    img_str = base64.b64encode(buff.getvalue())
    return img_str


# Convert Base64 to Image
def b64_2_img(data: str):
    from PIL import Image

    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

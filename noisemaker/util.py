import tensorflow as tf


def save(tensor, name="noise.png"):
    """
    Save an image Tensor to a file.

    :param Tensor tensor: Image tensor
    :param str name: Filename, ending with .png or .jpg
    :return: None
    """

    tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    if name.endswith(".png"):
        data = tf.image.encode_png(tensor).eval()

    elif name.endswith(".jpg"):
        data = tf.image.encode_jpeg(tensor).eval()

    else:
        raise ValueError("Filename should end with .png or .jpg")

    with open(name, "wb") as fh:
        fh.write(data)
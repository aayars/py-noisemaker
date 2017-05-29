import tensorflow as tf

# import noisemaker.effects as effects


def save(tensor, name="noise.png"):
    """
    Save as PNG. Prints a message to stdout.

    TODO: Support other image formats.

    :param Tensor tensor: Image tensor
    :param str name: Filename, ending with .png or .jpg
    :return: None
    """

    # tensor = effects.normalize(tensor)

    tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    if name.endswith(".png"):
        data = tf.image.encode_png(tensor).eval()

    elif name.endswith(".jpg"):
        data = tf.image.encode_jpeg(tensor).eval()

    else:
        raise ValueError("Filename should end with .png or .jpg")

    with open(name, "wb") as fh:
        fh.write(data)
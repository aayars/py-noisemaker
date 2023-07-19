"""Noise generation interface for Noisemaker"""

import base64
import os

import noisemaker.util as util

import requests
import tensorflow as tf

# Adapted from stability.ai API usage example
# https://platform.stability.ai/rest-api#tag/v1generation/operation/imageToImage

def apply(settings, seed, input_filename="art.png", output_filename="art-ai.png"):
    engine_id = "stable-diffusion-v1-5"
    # engine_id = "stable-diffusion-512-v2-1"
    api_host = "https://api.stability.ai"

    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.stability"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing Stability API key at {api_key_path}.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        files={
            "init_image": open(input_filename, "rb")
        },
        data={
            "image_strength": settings["image_strength"],
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": settings["prompt"],
            "cfg_scale": settings["cfg_scale"],
            "clip_guidance_preset": "FAST_BLUE",
            "samples": 1,
            "steps": 50,
            "seed": seed,
            "style_preset": settings["style_preset"],
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):
        with open(f"{output_filename}", "wb") as f:
            tensor = tf.io.decode_png(base64.b64decode(image["base64"]))

    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

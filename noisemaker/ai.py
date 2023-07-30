"""Noise generation interface for Noisemaker"""

#
# stability.ai key is needed for stable diffusion.
# Put the key in a file named: $NOISEMAKER_DIR/.stability
#
# OpenAI key is needed for alt text generation.
# Put the key in a file named: $NOISEMAKER_DIR/.openai
#
# $NOISEMAKER_DIR defaults to ~/.noisemaker
#

import base64
import os
import random
import time

import noisemaker.util as util

from colorthief import ColorThief

import requests
import tensorflow as tf


STABILITY_API_HOST = "https://api.stability.ai"
STABILITY_ENGINE_ID = "stable-diffusion-v1-5"

OPENAI_API_HOST = "https://api.openai.com"
OPENAI_MODEL = "gpt-3.5-turbo"


# Adapted from stability.ai API usage example
# https://platform.stability.ai/rest-api#tag/v1generation/operation/imageToImage
def apply(settings, seed, input_filename="art.png", output_filename="art-ai.png"):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.stability"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing Stability API key at {api_key_path}.")

    response = requests.post(
        f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/image-to-image",
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


def describe(name, prompt, filename):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.openai"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing OpenAI API key at {api_key_path}.")

    thief = ColorThief(filename)
    palette = thief.get_palette(color_count=random.randint(2,3))

    for color in reversed(palette):
        prompt = f"rgb({color[0]},{color[1]},{color[2]}), " + prompt

    response = requests.post(
        f"{OPENAI_API_HOST}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "I will provide a name of a generative art composition, along with a comma-delimited list of descriptive terms. You will use this information to generate a short summary, written in the authoritative tone of a fine art critic. Put quotes around the provided name of the composition, and properly capitalize it. Additionally, input may specify RGB color codes in the format of rgb(R,G,B) in the range of 0-255, but you must convert these into human-readable color names and refer to them as such. Do not refer to RGB color code representations or quote the names. Do not put the entire summary in quotes. The summary may not exceed 250 characters."
                },
                {
                    "role": "user",
	                "content": f"Create a human-readable English summary to be used as a descriptive \"alt text\" image caption, for those who are unable to see the image. The name of the composition is \"{name}\", and the list of terms is: \"{prompt}\""
                }
            ]
        }
    )

    summary = response.json()['choices'][0]['message']['content']

    response = requests.post(
        f"{OPENAI_API_HOST}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Sometimes the previous OpenAI assistant makes mistakes, it is your job to correct them. For the given summary of a generative art piece, perform the following fixes as necessary: Shorten the summary to 250 characters or fewer. Composition name must be in quotes and properly capitalized. Any lingering RGB color codes must be converted into human-readable color names. Color names must not be capitalized, nor in quotes. The summary paragraph must not be in quotes. Finally, check the grammar and tone of the summary, and make sure it doesn't sound too pretentious or repetitive."
                },
                {
                    "role": "user",
                    "content": summary
                }
            ]
        }
    )

    summary = response.json()['choices'][0]['message']['content']

    return summary
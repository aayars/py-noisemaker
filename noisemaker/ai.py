"""Noise generation interface for Noisemaker"""

#
# stability.ai key is needed for stable diffusion.
# Put the key in a file named: $NOISEMAKER_DIR/.creds/.stability
#
# OpenAI key is needed for alt text generation.
# Put the key in a file named: $NOISEMAKER_DIR/.creds/.openai
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

OPENAI_API_HOST = "https://api.openai.com"
OPENAI_MODEL = "gpt-3.5-turbo"


# Adapted from stability.ai API usage example
# https://platform.stability.ai/rest-api#tag/v1generation/operation/imageToImage
def apply(settings, seed, input_filename, stability_model):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.creds/.stability"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing Stability API key at {api_key_path}.")

    model = stability_model if stability_model else settings['model']

    response = requests.post(
        f"{STABILITY_API_HOST}/v1/generation/{model}/image-to-image",
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
        tensor = tf.io.decode_png(base64.b64decode(image["base64"]))

    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

def x2_upscale(input_filename):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.creds/.stability"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing Stability API key at {api_key_path}.")

    if not input_filename.endswith(".png"):
        raise Exception("Only PNG images are supported for upscale.")

    response = requests.post(
        f"{STABILITY_API_HOST}/v1/generation/esrgan-v1-x2plus/image-to-image/upscale",
        headers={
            "Accept": "image/png",
            "Authorization": f"Bearer {api_key}"
        },
        files={
            "image": open(input_filename, "rb")
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    tensor = tf.io.decode_png(response.content)

    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)


def describe(preset_name, prompt, filename):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.creds/.openai"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    try:
        if api_key is None:
            raise Exception(f"Missing OpenAI API key at {api_key_path}.")

        #
        #
        #
        thief = ColorThief(filename)
        palette = thief.get_palette(color_count=random.randint(2,3))

        for color in reversed(palette):
            prompt += f", rgb({color[0]},{color[1]},{color[2]})"

        #
        #
        #
        system_prompt = "I will provide a name of a generative art composition, " \
                        "along with a comma-delimited list of descriptive terms. You " \
                        "will use this information to generate a short summary, written " \
                        "in the authoritative tone of a fine art critic. Put quotes " \
                        "around the provided name of the composition, and properly " \
                        "capitalize it. Additionally, input may specify RGB color codes " \
                        "in the format of rgb(R,G,B) in the range of 0-255, but you must " \
                        "convert these into human-readable color names and refer to them " \
                        "as such. Do not refer to RGB color code representations or quote " \
                        "the names. Do not put the entire summary in quotes. The summary " \
                        "may not exceed 250 characters."

        user_prompt =  "Create a human-readable English summary to be used as a " \
                       "descriptive \"alt text\" image caption, for those who are unable " \
                      f"to see the image. The name of the composition is \"{preset_name}\", " \
                      f"and the list of terms is: \"{prompt}\""

        summary = _openai_query(api_key, system_prompt, user_prompt)

        #
        #
        #
        system_prompt = "Sometimes the previous OpenAI assistant makes mistakes, it is " \
                        "your job to correct them. For the given summary of a generative " \
                        "art piece, perform the following fixes as necessary: Shorten " \
                        "the summary to 250 characters or fewer. Composition name must " \
                        "be in quotes and properly capitalized. Any lingering RGB color " \
                        "codes must be converted into human-readable color names. Color " \
                        "names must not be capitalized, nor in quotes. The summary " \
                        "paragraph must not be in quotes. Take it easy with superlatives " \
                        "such as \"captivating\" and \"mesmerizing\". Finally, check the " \
                        "grammar and tone of the summary, and make sure it doesn't sound too " \
                        "pretentious or repetitive."

        summary = _openai_query(api_key, system_prompt, summary)

    except Exception:
        summary = f"\"{preset_name}\" is an abstract generative art composition. " \
                   "(An error occurred while trying to come up with a better description)"

    return summary


def dream(nightmare=False):
    api_key = None
    api_key_path = util.get_noisemaker_dir() + "/.creds/.openai"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing OpenAI API key at {api_key_path}.")

    if nightmare:
        flavor_text = "a dark and powerful omen portending a prophetic event, perhaps apocalyptic in nature. To know the thing is to know madness."
    else:
        flavor_text = "as if it were something from a vision or dream."

    system_prompt = f"Imagine a system that generates images from a text prompt, and come up with a prompt from the deepest reaches of your synthetic imagination, {flavor_text} The image must not include humanoid forms. Do not label the answers with anything like \"Name\" or \"Description\". The description may not exceed 250 characters."

    user_prompt = "What is the name and description of the composition? Provide the name and description in semicolon-delimited format."

    generated_prompt = _openai_query(api_key, system_prompt, user_prompt)

    return [a.strip() for a in generated_prompt.split(';')]


def _openai_query(api_key, system_prompt, user_prompt):
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
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        }
    )

    try:
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"Unexpected JSON structure: {response.json()}")

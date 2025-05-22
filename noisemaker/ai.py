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

import noisemaker.util as util

from colorthief import ColorThief

import requests
import tensorflow as tf


STABILITY_API_HOST = "https://api.stability.ai"

OPENAI_API_HOST = "https://api.openai.com"
OPENAI_MODEL = "o4-mini"


# Adapted from stability.ai API usage example
# https://platform.stability.ai/rest-api#tag/v1generation/operation/imageToImage
def apply(settings, seed, input_filename, stability_model):
    model = stability_model if stability_model else settings['model']

    if model in ('sd3', 'core', 'ultra'):
        return apply_v2(settings, seed, input_filename, stability_model)

    response = requests.post(
        f"{STABILITY_API_HOST}/v1/generation/{model}/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {_api_key('stability')}"
        },
        files={
            "init_image": open(input_filename, "rb")
        },
        data={
            "image_strength": settings["image_strength"],
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": settings["prompt"] + " No people.",
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


def apply_v2(settings, seed, input_filename, stability_model=None):
    model = stability_model if stability_model else settings['model']

    response = requests.post(
        f"{STABILITY_API_HOST}/v2beta/stable-image/generate/{model}",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {_api_key('stability')}"
        },
        files={
            "image": open(input_filename, "rb")
        },
        data={
            "mode": "image-to-image",
            "prompt": settings["prompt"],
            "negative_prompt": settings.get("negative_prompt", "People, Words"),
            "strength": settings["image_strength"],
            "seed": seed,
            "output_format": "png"
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + response.text)

    result = response.json()

    if result.get("finish_reason") != "SUCCESS":
        raise Exception(result.get("finish_reason", str(result)))

    image_b64 = result.get("image")
    if not image_b64:
        raise Exception("Image data not found in the response.")

    tensor = tf.io.decode_png(base64.b64decode(image_b64))
    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)


def apply_style(settings, seed, content_filename, style_filename, output_format="png"):
    response = requests.post(
        f"{STABILITY_API_HOST}/v2beta/stable-image/control/style-transfer",
        headers={
            "Accept": "image/*",
            'Authorization': f"Bearer {_api_key('stability')}"
        },
        files={
            "init_image": open(content_filename, "rb"),
            "style_image": open(style_filename, "rb")
        },
        data={
            "prompt": settings["prompt"],
            "negative_prompt": settings.get("negative_prompt", "People, Words"),
            "style_strength": settings["image_strength"],
            "seed": seed,
            "output_format": output_format,
        }
    )

    if response.status_code != 200:
        raise Exception('Non-200 response: ' + response.text)

    tensor = tf.io.decode_image(response.content, channels=0)

    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)


def x4_upscale(input_filename):
    response = requests.post(
        f"{STABILITY_API_HOST}/v2beta/stable-image/upscale/fast",
        headers={
            "Accept": "image/*",
            "Authorization": f"Bearer {_api_key('stability')}"
        },
        files={
            "image": open(input_filename, "rb")
        },
        data={
            "output_format": "png"
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + response.text)

    tensor = tf.io.decode_png(response.content)
    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)


def describe(preset_name, prompt, filename):
    try:
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
        system_prompt = """
	        You will be provided with a name of a generative art
	        composition, along with a comma-delimited list of
	        descriptive terms. You will use this information to
	        generate a brief but informative alt text caption.  Put
	        quotes around the provided name of the composition, and
	        properly capitalize it.  Additionally, input may specify
	        RGB color codes in the format of rgb(R,G,B) in the range
	        of 0-255, but you must convert these into human-readable
	        color names and refer to them as such. Do not refer to
	        RGB color code representations or quote the names.  Do
	        not put the entire summary in quotes.  The summary may
	        not exceed 250 characters.
                        """

        user_prompt =  f"""
	        Create a human-readable English summary to be used as
	        a descriptive \"alt text\" image caption, for those who
	        are unable to see the image. The name of the composition
	        is \"{preset_name}\", and the list of terms is:
	        \"{prompt}\"
                      """

        summary = _openai_query(system_prompt, user_prompt)

        #
        #
        #
        system_prompt = """
	        Sometimes the previous OpenAI assistant makes mistakes, it
	        is your job to correct them. For the given summary of a
	        generative art piece, perform the following fixes as
	        necessary: Shorten the summary to 250 characters or fewer.
	        Composition name must be in quotes and properly capitalized.
	        Any lingering RGB color codes must be converted into
	        human-readable color names. Color names must be lower-case,
	        and not in quotes. The summary paragraph must not be in
	        quotes. Take it easy with superlatives such as \"captivating\"
	        and \"mesmerizing\". Finally, check the grammar and tone
	        of the summary, and make sure it doesn't sound too pretentious
	        or repetitive.
                        """

        summary = _openai_query(system_prompt, summary)

    except Exception:
        summary = f"\"{preset_name}\" is an abstract generative art composition. " \
                   "(An error occurred while trying to come up with a better description)"

    return summary


def _api_key(api):
    api_key = None
    api_key_path = f"{util.get_noisemaker_dir()}/.creds/.{api}"
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as fh:
            api_key = fh.read().strip()

    if api_key is None:
        raise Exception(f"Missing {api} API key at {api_key_path}.")

    return api_key


def _openai_query(system_prompt, user_prompt):
    response = requests.post(
        f"{OPENAI_API_HOST}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {_api_key('openai')}",
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
            ],
        }
    )

    try:
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"Unexpected JSON structure: {response.json()}")

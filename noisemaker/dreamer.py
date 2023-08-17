"""Dynamic prompt and image generator"""

import random
import tempfile
import time

from noisemaker.constants import ColorSpace

import noisemaker.ai as ai
import noisemaker.composer as composer
import noisemaker.generators as generators
import noisemaker.util as util


_ADJECTIVES = [
    "artistic",
    "conceptual",
    "fantasy",
    "futuristic",
    "magical",
    "minimalist",
    "modern",
    "peaceful",
    "photographic",
    "psychedelic",
    "realistic",
    "restful",
    "retro",
    "rustic",
    "sci-fi",
    "soothing",
    "visionary",
    "vintage",
    "whimsical",
]

_INTROS = [
    "I'm having a dream about...",
    "Last night I dreamed...",
    "I was just reminded of a dream I had about...",
    "I had a daydream where...",
    "I vaguely recall dreaming about...",
    "When I close my eyes, I can picture...",
    "I keep having a recurring dream about...",
    "I dream of...",
    "I used to dream about...",
    "I had a very realistic dream about...",
    "I had a bizarre dream where...",
    "I sometimes imagine...",
    "I sometimes dream of...",
    "I just remembered a recent dream where...",
]

def dream(width, height, filename='dream.png'):
    adjective = composer.random_member(_ADJECTIVES)

    system_prompt = f"Imagine a system that generates images from a text prompt, and come up with a prompt for an image in a {adjective} style. Provide the data in semicolon-delimited format. Do not litter the data with useless labels like \"Name:\" or \"Description:\" or \"the name is\" or \"the description is\" or \"the name and description are as follows:\". Properly capitalize and use spaces where appropriate. The name must be between 8 and 25 characters. The description may be between 50 and 225 characters."

    user_prompt = "What is the name and description of the composition? Provide the data in semicolon-delimited format. Do not label the data."

    generated_prompt = ai._openai_query(system_prompt, user_prompt)

    # Sigh
    if len(generated_prompt.split(';')) != 2 or any(string in generated_prompt.lower() for string in [':', '_', '"', 'name', 'description']):
        time.sleep(1)

        system_prompt = f"ChatGPT is not very good at following directions. This data is supposed to contain only a semicolon-delimited name and description. Ensure that the given data is not littered with useless labels like \"Name:\" or \"Description:\" or \"the name is\" or \"the description is\" or \"the name and description are as follows:\". The generated name and description should be seperated by a semicolon. Properly capitalize and use spaces where appropriate. The name must be between 8 and 25 characters. The description may be between 50 and 225 characters."

        generated_prompt = ai._openai_query(system_prompt, generated_prompt)

    name, prompt = [a.strip() for a in generated_prompt.split(';')]

    intro = composer.random_member(_INTROS)

    system_prompt = f"Modify the received prompt to indicate that it's something from a dream. You may begin the statement by paraphrasing that it was from a dream. For example, come up with a variation for something like: \"{intro}\". The statement may not exceed 250 characters."

    message = ai._openai_query(system_prompt, prompt)

    shape = [height, width, 3]
    color_space = composer.random_member([m for m in ColorSpace if m != ColorSpace.grayscale])

    tensor = generators.basic(freq=[height, width], shape=shape, color_space=color_space,
                              hue_range=0.125 + random.random() * 1.5)

    settings = {
        "image_strength": 0.0125,
        "cfg_scale": 20,
        "prompt": prompt,
        "style_preset": "photographic",
        "model": "stable-diffusion-xl-1024-v1-0",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_filename = f"{tmp}/noise.png"

        util.save(tensor, tmp_filename)

        tensor = ai.apply(settings, random.randint(1, 2 ** 32 - 1), tmp_filename, None)

        description = ai.describe(name, prompt, tmp_filename)

    composer.EFFECT_PRESETS["lens"].render(seed=random.randint(1, 2 ** 32 - 1), tensor=tensor, shape=shape, filename=filename)

    return name, message, description

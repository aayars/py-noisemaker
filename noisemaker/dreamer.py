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
    "abstract",
    "art deco",
    "art noveau",
    "artistic",
    "conceptual",
    "contemporary",
    "cubist",
    "expressionist",
    "fantasy",
    "futuristic",
    "impressionist",
    "macro",
    "magical",
    "minimalist",
    "modern",
    "peaceful",
    "photographic",
    "post-impressionist",
    "psychedelic",
    "realistic",
    "restful",
    "retro",
    "rustic",
    "sci-fi",
    "soothing",
    "still life",
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

    for _ in range(5):
        system_prompt = f"Imagine a system that generates images from a text prompt, and come up with a prompt for an image in a {adjective} style. Your answer is intended to be machine-readable, so do not litter the answers with labels like \"Name\" or \"Description\" or \"the name is\" or \"the description is\" or \"the name and description are as follows\". The description may not exceed 250 characters."

        user_prompt = "What is the name and description of the composition? Provide the name and description in semicolon-delimited format."

        generated_prompt = ai._openai_query(system_prompt, user_prompt)

        # Brute force it when GPT doesn't follow directions
        if len(generated_prompt.split(';')) == 2 and not any(string in generated_prompt.lower() for string in ['_', '"', 'name', 'description']):
            break

        time.sleep(1)

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

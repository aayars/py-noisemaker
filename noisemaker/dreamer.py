"""Dynamic prompt and image generator"""

import random
import tempfile
import time

from noisemaker.constants import ColorSpace

import noisemaker.ai as ai
import noisemaker.composer as composer
import noisemaker.generators as generators
import noisemaker.util as util
import noisemaker.value as value


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
    "Imagine yourself standing in...",
    "Imagine what it must be like to...",
    "You find yourself in...",
    "A scene appears in your mind...",
    "Picture in your mind...",
    "Imagine a real place in your mind...",
    "Close your eyes and try to picture...",
    "Imagine going to...",
    "Imagine the sights, sounds, and smells of...",
]

def dream(width, height, filename='dream.png'):
    adjective = composer.random_member(_ADJECTIVES)

    system_prompt = f"Imagine a system that generates images from a text prompt, and come up with a prompt for a natural scene rendered in a {adjective} style. Provide the data in semicolon-delimited format. Do not litter the data with useless labels like \"Name:\" or \"Description:\" or \"the name is\" or \"the description is\" or \"the name and description are as follows:\". Properly capitalize and use spaces where appropriate. The name must be between 8 and 25 characters. The description may be between 50 and 225 characters."

    user_prompt = "What is the name and description of the composition? Provide the data in semicolon-delimited format. Do not label the data."

    generated_prompt = ai._openai_query(system_prompt, user_prompt)

    # Sigh
    if len(generated_prompt.split(';')) != 2 or any(string in generated_prompt.lower() for string in [':', '_', '"', 'name', 'description']):
        time.sleep(1)

        system_prompt = f"ChatGPT is not very good at following directions. This data is supposed to contain only a semicolon-delimited name and description. Ensure that the given data is not littered with useless labels like \"Name:\" or \"Description:\" or \"the name is\" or \"the description is\" or \"the name and description are as follows:\". The generated name and description should be seperated by a semicolon. Properly capitalize and use spaces where appropriate. The name must be between 8 and 25 characters. The description may be between 50 and 225 characters."

        generated_prompt = ai._openai_query(system_prompt, generated_prompt)

    name, prompt = [a.strip() for a in generated_prompt.split(';')]

    intro = composer.random_member(_INTROS)

    system_prompt = f"Modify the received prompt as a visualization aid. You may begin the statement by paraphrasing that the viewer should mentally transport themselves into this scene. For example, come up with a variation for something like: \"{intro}\". The statement may not exceed 250 characters."

    message = ai._openai_query(system_prompt, prompt)

    shape = [height, width, 3]
    color_space = composer.random_member([m for m in ColorSpace if m != ColorSpace.grayscale])

    seed = random.randint(1, 99999999)

    value.set_seed(seed)

    tensor = generators.basic(freq=[int(height * .5), int(width * .5)], shape=shape, color_space=color_space,
                              hue_range=0.125 + random.random() * 1.5)

    new_tensor = generators.basic(freq=[int(height * .005), int(width * .005)], shape=shape, color_space=color_space,
                                  hue_range=0.125 + random.random() * 1.0)

    tensor = value.blend(tensor, new_tensor, 0.5)

    settings = {
        "image_strength": 0.0125,
        "seed": seed,
        "cfg_scale": 20,
        "prompt": prompt,
        "style_preset": "photographic",
        "model": "core",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_filename = f"{tmp}/noise.png"
        # tmp_filename = f"input.png"  # XXX

        util.save(tensor, tmp_filename)

        style_tensor = ai.apply(settings, random.randint(1, 2 ** 32 - 1), tmp_filename, None)
        style_tensor = value.resample(style_tensor, shape)

        style_filename = f"{tmp}/temp-style.png"
        # style_filename = "style.png"  # XXX

        util.save(style_tensor, style_filename)

        new_tensor = ai.apply_style(settings, seed, tmp_filename, style_filename)

        tensor = value.blend(new_tensor, tensor, 0.1)

        description = ai.describe(name, prompt, tmp_filename)

    composer.EFFECT_PRESETS["lens"].render(seed=random.randint(1, 2 ** 32 - 1), tensor=tensor, shape=shape, filename=filename)

    util.save(ai.x4_upscale(filename), filename)

    return name, message, description

import SimpleComputeEffect from '../../common/simple-compute-effect.js';
import metadata from './meta.json' with { type: 'json' };

const GLYPH_WIDTH = 16;
const GLYPH_HEIGHT = 16;
const GLYPH_COUNT = 8;
const GLYPH_ATLAS_WIDTH = GLYPH_WIDTH;
const GLYPH_ATLAS_HEIGHT = GLYPH_HEIGHT * GLYPH_COUNT;
const GLYPH_BYTES_PER_ROW = GLYPH_ATLAS_WIDTH * 4 * Float32Array.BYTES_PER_ELEMENT;

function generateGlyphAtlas() {
  const totalPixels = GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT;
  const data = new Float32Array(totalPixels * 4);
  const mid = (GLYPH_WIDTH - 1) * 0.5;

  for (let glyphIndex = 0; glyphIndex < GLYPH_COUNT; glyphIndex += 1) {
    const glyphOffsetY = glyphIndex * GLYPH_HEIGHT;
    for (let y = 0; y < GLYPH_HEIGHT; y += 1) {
      for (let x = 0; x < GLYPH_WIDTH; x += 1) {
        const atlasY = glyphOffsetY + y;
        const atlasIndex = (atlasY * GLYPH_ATLAS_WIDTH + x) * 4;
        const dx = x - mid;
        const dy = y - mid;
        const distance = Math.sqrt(dx * dx + dy * dy);
        let value = 0;

        switch (glyphIndex) {
          case 0: // solid block
            value = 1;
            break;
          case 1: // horizontal stripes
            value = (y % 4) < 2 ? 1 : 0;
            break;
          case 2: // vertical stripes
            value = (x % 4) < 2 ? 1 : 0;
            break;
          case 3: // diagonal cross
            value = (Math.abs(x - y) <= 1 || Math.abs((x + y) - (GLYPH_WIDTH - 1)) <= 1) ? 1 : 0;
            break;
          case 4: // hollow square
            value = (x <= 1 || x >= GLYPH_WIDTH - 2 || y <= 1 || y >= GLYPH_HEIGHT - 2) ? 1 : 0;
            break;
          case 5: // plus sign
            value = (Math.abs(x - mid) <= 1 || Math.abs(y - mid) <= 1) ? 1 : 0;
            break;
          case 6: // circle rim
            value = (distance > mid - 3 && distance < mid - 1) ? 1 : 0;
            break;
          default: // checkerboard
            value = ((x + y) % 2 === 0) ? 1 : 0;
            break;
        }

        const normalized = value ? 1 : 0;
        data[atlasIndex + 0] = normalized;
        data[atlasIndex + 1] = normalized;
        data[atlasIndex + 2] = normalized;
        data[atlasIndex + 3] = 1;
      }
    }
  }

  return data;
}

class GlyphMapEffect extends SimpleComputeEffect {
  static metadata = metadata;

  constructor(options = {}) {
    super(options);
    this.glyphTexture = null;
    this.glyphTextureDevice = null;
    this.glyphTextureDirty = true;
    this.glyphAtlasData = generateGlyphAtlas();
  }

  destroy() {
    this.#destroyGlyphTexture();
    super.destroy();
  }

  getResourceCreationOptions(context = {}) {
    const base = super.getResourceCreationOptions(context) ?? {};
    const inputTextures = { ...(base.inputTextures ?? {}) };
    const glyphTexture = this.#ensureGlyphTexture(context);
    if (glyphTexture) {
      inputTextures.glyph_texture = glyphTexture;
    }

    return {
      ...base,
      inputTextures,
    };
  }

  async onResourcesCreated(resources, context) {
    const baseResources = await super.onResourcesCreated(resources, context);
    const finalResources = baseResources ?? resources;
    if (!finalResources) {
      return baseResources;
    }

    const { paramsState, bindingOffsets } = finalResources;
    if (paramsState && bindingOffsets) {
      let updated = false;
  updated = this.#writeDefaultParam(paramsState, bindingOffsets, 'channels', 4) || updated;
  updated = this.#writeDefaultParam(paramsState, bindingOffsets, 'glyph_width', GLYPH_WIDTH) || updated;
      updated = this.#writeDefaultParam(paramsState, bindingOffsets, 'glyph_height', GLYPH_HEIGHT) || updated;
      updated = this.#writeDefaultParam(paramsState, bindingOffsets, 'glyph_count', GLYPH_COUNT) || updated;
      if (updated) {
        finalResources.paramsDirty = true;
      }
    }

    return finalResources;
  }

  #writeDefaultParam(paramsState, offsets, key, value) {
    const offset = offsets?.[key];
    if (!Number.isInteger(offset) || offset < 0 || offset >= paramsState.length) {
      return false;
    }
    if (paramsState[offset] === value) {
      return false;
    }
    paramsState[offset] = value;
    return true;
  }

  #ensureGlyphTexture(context = {}) {
    const device = context?.device;
    if (!device) {
      return this.glyphTexture;
    }

    const deviceChanged = this.glyphTextureDevice && this.glyphTextureDevice !== device;
    if (!this.glyphTexture || deviceChanged) {
      this.#destroyGlyphTexture();
      try {
        this.glyphTexture = device.createTexture({
          size: { width: GLYPH_ATLAS_WIDTH, height: GLYPH_ATLAS_HEIGHT, depthOrArrayLayers: 1 },
          format: 'rgba32float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.glyphTextureDevice = device;
        this.glyphTextureDirty = true;
      } catch (error) {
        this.helpers.logWarn?.('GlyphMap: failed to create glyph texture.', error);
        this.glyphTexture = null;
        this.glyphTextureDevice = null;
        this.glyphTextureDirty = true;
        return null;
      }
    }

    if (this.glyphTextureDirty && this.glyphTexture) {
      this.#uploadGlyphAtlas(device);
    }

    return this.glyphTexture;
  }

  #uploadGlyphAtlas(device) {
    if (!this.glyphTexture || !device) {
      return;
    }

    try {
      device.queue.writeTexture(
        { texture: this.glyphTexture },
        this.glyphAtlasData,
        {
          bytesPerRow: GLYPH_BYTES_PER_ROW,
          rowsPerImage: GLYPH_ATLAS_HEIGHT,
        },
        {
          width: GLYPH_ATLAS_WIDTH,
          height: GLYPH_ATLAS_HEIGHT,
          depthOrArrayLayers: 1,
        },
      );
      this.glyphTextureDirty = false;
    } catch (error) {
      this.helpers.logWarn?.('GlyphMap: failed to upload glyph atlas.', error);
      this.glyphTextureDirty = true;
    }
  }

  #destroyGlyphTexture() {
    if (this.glyphTexture?.destroy) {
      try {
        this.glyphTexture.destroy();
      } catch (error) {
        this.helpers.logWarn?.('GlyphMap: failed to destroy glyph texture.', error);
      }
    }
    this.glyphTexture = null;
    this.glyphTextureDevice = null;
    this.glyphTextureDirty = true;
  }
}

export default GlyphMapEffect;

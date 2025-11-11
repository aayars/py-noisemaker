// Spooky ticker effect that renders flickering segmented glyphs crawling across the image.
// Mirrors the behaviour of noisemaker.effects.spooky_ticker.

const CHANNEL_COUNT : u32 = 4u;
const INV_U32_MAX : f32 = 1.0 / 4294967295.0;

struct SpookyTickerParams {
    width : f32,
    height : f32,
    channels : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SpookyTickerParams;

const MASK_ARECIBO_NUCLEOTIDE : u32 = 0u;
const MASK_ARECIBO_NUM : u32 = 1u;
const MASK_BANK_OCR : u32 = 2u;
const MASK_BAR_CODE : u32 = 3u;
const MASK_BAR_CODE_SHORT : u32 = 4u;
const MASK_EMOJI : u32 = 5u;
const MASK_FAT_LCD_HEX : u32 = 6u;
const MASK_ALPHANUM_HEX : u32 = 7u;
const MASK_ICHING : u32 = 8u;
const MASK_IDEOGRAM : u32 = 9u;
const MASK_INVADERS : u32 = 10u;
const MASK_LCD : u32 = 11u;
const MASK_LETTERS : u32 = 12u;
const MASK_MATRIX : u32 = 13u;
const MASK_ALPHANUM_NUMERIC : u32 = 14u;
const MASK_SCRIPT : u32 = 15u;
const MASK_WHITE_BEAR : u32 = 16u;

const MASK_CHOICES : array<u32, 17> = array<u32, 17>(
    MASK_ARECIBO_NUCLEOTIDE,
    MASK_ARECIBO_NUM,
    MASK_BANK_OCR,
    MASK_BAR_CODE,
    MASK_BAR_CODE_SHORT,
    MASK_EMOJI,
    MASK_FAT_LCD_HEX,
    MASK_ALPHANUM_HEX,
    MASK_ICHING,
    MASK_IDEOGRAM,
    MASK_INVADERS,
    MASK_LCD,
    MASK_LETTERS,
    MASK_MATRIX,
    MASK_ALPHANUM_NUMERIC,
    MASK_SCRIPT,
    MASK_WHITE_BEAR
);
const MASK_CHOICES_COUNT : u32 = 17u;

const HEX_SEGMENTS : array<u32, 16> = array<u32, 16>(
    0x77u, // 0b1110111
    0x24u, // 0b0100100
    0x5Du, // 0b1011101
    0x6Du, // 0b1101101
    0x2Eu, // 0b0101110
    0x6Bu, // 0b1101011
    0x7Bu, // 0b1111011
    0x25u, // 0b0100101
    0x7Fu, // 0b1111111
    0x6Fu, // 0b1101111
    0x3Fu, // 0b0111111
    0x7Au, // 0b1111010
    0x53u, // 0b1010011
    0x7Cu, // 0b1111100
    0x5Bu, // 0b1011011
    0x1Bu  // 0b0011011
);

struct MaskShape {
    height : i32,
    width : i32,
};

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_COUNT)) {
        return CHANNEL_COUNT;
    }
    return u32(rounded);
}

fn pixel_base_index(x : u32, y : u32, width : u32) -> u32 {
    return (y * width + x) * CHANNEL_COUNT;
}

fn clamp_i32(value : i32, lo : i32, hi : i32) -> i32 {
    var min_v : i32 = lo;
    var max_v : i32 = hi;
    if (min_v > max_v) {
        let tmp : i32 = min_v;
        min_v = max_v;
        max_v = tmp;
    }
    if (value < min_v) {
        return min_v;
    }
    if (value > max_v) {
        return max_v;
    }
    return value;
}

fn wrap_coord(value : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    var wrapped : i32 = value % size;
    if (wrapped < 0) {
        wrapped = wrapped + size;
    }
    return wrapped;
}

fn hash_mix(value : u32) -> u32 {
    var v : u32 = value;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn combine_seed(base : u32, salt : u32) -> u32 {
    return hash_mix(base ^ (salt * 0x9e3779b9u + 0x85ebca6bu));
}

fn random_u32(base : u32, salt : u32) -> u32 {
    return hash_mix(base ^ salt);
}

fn random_float(base : u32, salt : u32) -> f32 {
    return f32(random_u32(base, salt)) * INV_U32_MAX;
}

fn random_int_inclusive(base : u32, salt : u32, lo : i32, hi : i32) -> i32 {
    if (hi <= lo) {
        return lo;
    }
    let span : u32 = u32(hi - lo + 1);
    if (span == 0u) {
        return lo;
    }
    let value : u32 = random_u32(base, salt);
    return lo + i32(value % span);
}

fn lerp_f32(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
}

fn mask_noise(seed : u32, x : i32, y : i32, salt : u32) -> f32 {
    let packed : u32 = (u32(x & 0xffff) << 16) ^ u32(y & 0xffff) ^ (salt * 0x45d9f3b8u);
    return random_float(seed, packed);
}

fn digital_segment_value(
    local_x : i32,
    local_y : i32,
    width : i32,
    height : i32,
    pattern : u32,
    thickness : f32,
) -> f32 {
    if (width <= 0 || height <= 0) {
        return 0.0;
    }

    let w : f32 = f32(width);
    let h : f32 = f32(height);
    let stroke_h : i32 = max(1, i32(round(h * thickness)));
    let stroke_v : i32 = max(1, i32(round(w * thickness * 0.6)));

    let mid_start : i32 = clamp_i32(i32(floor(h * 0.5)) - stroke_h / 2, 1, height - 1);
    let mid_end : i32 = clamp_i32(mid_start + stroke_h, mid_start + 1, height);
    let top_limit : i32 = stroke_h;
    let bottom_start : i32 = height - stroke_h;

    var on : bool = false;

    if ((pattern & (1u << 0u)) != 0u) {
        on = on || (local_y < top_limit);
    }
    if ((pattern & (1u << 6u)) != 0u) {
        on = on || (local_y >= bottom_start);
    }
    if ((pattern & (1u << 3u)) != 0u) {
        on = on || (local_y >= mid_start && local_y < mid_end);
    }

    if ((pattern & (1u << 1u)) != 0u) {
        on = on || (local_x < stroke_v && local_y >= top_limit && local_y < mid_start);
    }
    if ((pattern & (1u << 4u)) != 0u) {
        on = on || (local_x < stroke_v && local_y >= mid_end && local_y < bottom_start);
    }
    if ((pattern & (1u << 2u)) != 0u) {
        on = on || (local_x >= width - stroke_v && local_y >= top_limit && local_y < mid_start);
    }
    if ((pattern & (1u << 5u)) != 0u) {
        on = on || (local_x >= width - stroke_v && local_y >= mid_end && local_y < bottom_start);
    }

    return select(0.0, 1.0, on);
}

fn mask_shape(mask_type : u32, seed : u32) -> MaskShape {
    switch mask_type {
        case MASK_ARECIBO_NUCLEOTIDE: {
            return MaskShape(6, 6);
        }
        case MASK_ARECIBO_NUM: {
            return MaskShape(6, 3);
        }
        case MASK_BANK_OCR: {
            return MaskShape(8, 7);
        }
        case MASK_BAR_CODE: {
            return MaskShape(24, 1);
        }
        case MASK_BAR_CODE_SHORT: {
            return MaskShape(10, 1);
        }
        case MASK_EMOJI: {
            return MaskShape(13, 13);
        }
        case MASK_FAT_LCD_HEX: {
            return MaskShape(10, 10);
        }
        case MASK_ALPHANUM_HEX: {
            return MaskShape(6, 6);
        }
        case MASK_ICHING: {
            return MaskShape(14, 8);
        }
        case MASK_IDEOGRAM: {
            let size : i32 = random_int_inclusive(seed, 3u, 4, 6) * 2;
            return MaskShape(size, size);
        }
        case MASK_INVADERS: {
            let h : i32 = random_int_inclusive(seed, 5u, 5, 7);
            let w : i32 = random_int_inclusive(seed, 7u, 6, 12);
            return MaskShape(h, w);
        }
        case MASK_LCD: {
            return MaskShape(8, 5);
        }
        case MASK_LETTERS: {
            let height : i32 = random_int_inclusive(seed, 11u, 3, 4) * 2 + 1;
            let width : i32 = random_int_inclusive(seed, 13u, 3, 4) * 2 + 1;
            return MaskShape(height, width);
        }
        case MASK_MATRIX: {
            return MaskShape(6, 4);
        }
        case MASK_ALPHANUM_NUMERIC: {
            return MaskShape(6, 6);
        }
        case MASK_SCRIPT: {
            let h : i32 = random_int_inclusive(seed, 13u, 7, 9);
            let w : i32 = random_int_inclusive(seed, 17u, 12, 24);
            return MaskShape(h, w);
        }
        case MASK_WHITE_BEAR: {
            return MaskShape(4, 4);
        }
        default: {
            return MaskShape(6, 6);
        }
    }
}

fn mask_multiplier(mask_type : u32, mask_width : i32) -> i32 {
    // Uniform larger multiplier for all rows
    return 8;
}

fn mask_padding(mask_type : u32, glyph_width : i32) -> i32 {
    if (glyph_width <= 0) {
        return 0;
    }
    if (mask_type == MASK_BAR_CODE || mask_type == MASK_BAR_CODE_SHORT) {
        return 0;
    }
    // Fixed single trailing pixel gap to mirror Python kerning behaviour.
    return 1;
}

fn digital_pattern(mask_type : u32, glyph_seed : u32) -> u32 {
    if (mask_type == MASK_ALPHANUM_HEX || mask_type == MASK_FAT_LCD_HEX) {
        let index : u32 = random_u32(glyph_seed, 23u) % 16u;
        return HEX_SEGMENTS[index];
    }
    let digit_index : u32 = random_u32(glyph_seed, 29u) % 10u;
    return HEX_SEGMENTS[digit_index];
}

fn sample_mask_pattern(
    mask_type : u32,
    local_x : i32,
    local_y : i32,
    mask_width : i32,
    mask_height : i32,
    glyph_seed : u32,
    row_seed : u32,
) -> f32 {
    if (mask_width <= 0 || mask_height <= 0) {
        return 0.0;
    }

    switch mask_type {
        case MASK_ARECIBO_NUCLEOTIDE: {
            if (local_y == 0 || local_y == mask_height - 1 || local_x == 0) {
                return 0.0;
            }
            if (local_y == mask_height - 2) {
                return 1.0;
            }
            if (local_y < mask_height - 3 && local_x > mask_width - 2) {
                return 0.0;
            }
            return select(0.0, 1.0, mask_noise(glyph_seed, local_x, local_y, 3u) < 0.5);
        }
        case MASK_ARECIBO_NUM: {
            if (local_y == 0 || local_y == mask_height - 1 || local_x == 0) {
                return 0.0;
            }
            if (local_y == mask_height - 2) {
                return select(0.0, 1.0, local_x == 1);
            }
            return select(0.0, 1.0, mask_noise(glyph_seed, local_x, local_y, 5u) < 0.5);
        }
        case MASK_BANK_OCR, MASK_FAT_LCD_HEX, MASK_ALPHANUM_HEX, MASK_ALPHANUM_NUMERIC, MASK_LCD: {
            let pattern : u32 = digital_pattern(mask_type, glyph_seed);
            let thickness : f32 = select(0.12, 0.18, mask_type == MASK_FAT_LCD_HEX);
            return digital_segment_value(local_x, local_y, mask_width, mask_height, pattern, thickness);
        }
        case MASK_BAR_CODE: {
            return select(0.0, 1.0, mask_noise(row_seed, local_x, 0, 7u) < 0.6);
        }
        case MASK_BAR_CODE_SHORT: {
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 11u) < 0.55);
        }
        case MASK_EMOJI: {
            let cx : f32 = (f32(mask_width) - 1.0) * 0.5;
            let cy : f32 = (f32(mask_height) - 1.0) * 0.5;
            let dx : f32 = (f32(local_x) + 0.5) - cx;
            let dy : f32 = (f32(local_y) + 0.5) - cy;
            let radius : f32 = min(cx, cy) * 0.8;
            let dist : f32 = sqrt(dx * dx + dy * dy);
            if (dist > radius) {
                return select(0.0, 1.0, mask_noise(glyph_seed, local_x, local_y, 19u) > 0.85);
            }
            let base : f32 = 0.5 + mask_noise(glyph_seed, local_x, local_y, 23u) * 0.5;
            return clamp(base, 0.0, 1.0);
        }
        case MASK_ICHING: {
            if (local_x == 0 || local_y == 0 || local_x == mask_width - 1 || local_y == mask_height - 1) {
                return 0.0;
            }
            if ((local_y % 2) == 0) {
                return 0.0;
            }
            if ((local_x % 2) == 1 && local_x % mask_width != 3 && local_x % mask_width != 4) {
                return 1.0;
            }
            if ((local_x % 2) == 0) {
                return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 31u) < 0.5);
            }
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 37u) < 0.45);
        }
        case MASK_IDEOGRAM: {
            if (local_x == 0 || local_y == 0 || local_x == mask_width - 1 || local_y == mask_height - 1) {
                return 0.0;
            }
            if (((local_x % 2) == 1) && ((local_y % 2) == 1)) {
                return 0.0;
            }
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 41u) > 0.5);
        }
        case MASK_INVADERS, MASK_WHITE_BEAR: {
            if (local_x == 0 || local_y == 0) {
                return 0.0;
            }
            let half_width : i32 = (mask_width + 1) / 2;
            var sample_x : i32 = local_x;
            if (local_x >= half_width) {
                sample_x = mask_width - 1 - local_x;
            }
            return select(0.0, 1.0, mask_noise(glyph_seed, sample_x, local_y, 43u) < 0.5);
        }
        case MASK_MATRIX: {
            if (local_x == 0 || local_y == 0) {
                return 0.0;
            }
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 47u) < 0.5);
        }
        case MASK_LETTERS: {
            if (local_x == 0 || local_y == 0 || local_x == mask_width - 1 || local_y == mask_height - 1) {
                return 0.0;
            }
            if ((mask_width - 1 - local_x == 0) || (mask_height - 1 - local_y == 0)) {
                return 0.0;
            }
            if ((local_x % 2 == 0) && (local_y % 2 == 0)) {
                return 0.0;
            }
            if ((local_x % 2 == 0) || (local_y % 2 == 0)) {
                return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 53u) > 0.25);
            }
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 59u) > 0.75);
        }
        case MASK_SCRIPT: {
            if (local_y == 0 || local_x == 0 || local_y == mask_height - 1) {
                return 0.0;
            }
            let step_y : i32 = local_y % mask_height;
            if (step_y == 1 || step_y == 3 || step_y == 6) {
                return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 61u) > 0.25);
            }
            if (step_y == 2 || step_y == 4 || step_y == 5) {
                return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 67u) > 0.9);
            }
            if ((mask_width - local_x) <= 1) {
                return 0.0;
            }
            if ((mask_height - step_y) <= 1) {
                return 0.0;
            }
            if ((local_x % 2 == 0) && (step_y % 2 == 0)) {
                return 0.0;
            }
            return select(0.0, 1.0, mask_noise(row_seed, local_x, local_y, 71u) > 0.5);
        }
        default: {
            return select(0.0, 1.0, mask_noise(glyph_seed, local_x, local_y, 73u) < 0.5);
        }
    }
}

fn sample_padded_mask(
    mask_type : u32,
    padded_x : i32,
    mask_y : i32,
    glyph_width : i32,
    mask_height : i32,
    glyph_padding : i32,
    glyph_seed : u32,
    row_seed : u32,
) -> f32 {
    if (glyph_width <= 0) {
        return 0.0;
    }
    if (padded_x < 0) {
        return 0.0;
    }
    if (glyph_padding > 0 && padded_x >= glyph_width && padded_x < glyph_width + glyph_padding) {
        return 0.0;
    }
    if (padded_x >= glyph_width + glyph_padding) {
        return 0.0;
    }
    return sample_mask_pattern(mask_type, padded_x, mask_y, glyph_width, mask_height, glyph_seed, row_seed);
}

fn ticker_mask(
    coord_x : i32,
    coord_y : i32,
    width : i32,
    height : i32,
    base_seed : u32,
    time_value : f32,
    speed_value : f32,
) -> f32 {
    if (width <= 0 || height <= 0) {
        return 0.0;
    }

    // Force maximum rows (3) for denser ticker. Previously random 1-3.
    let row_count : i32 = 3;
    var bottom_padding : i32 = 2;
    var mask_value : f32 = 0.0;

    // Simple approach: each row uses its natural mask height * multiplier, no artificial scaling
    var row_heights : array<i32, 3>;

    var row_index : i32 = 0;
    loop {
        if (row_index >= row_count) {
            break;
        }

        let row_seed : u32 = combine_seed(base_seed, u32(row_index) * 97u + 13u);
        let mask_choice_index : i32 = random_int_inclusive(row_seed, 109u, 0, 16);
        let mask_type : u32 = MASK_CHOICES[u32(mask_choice_index) % MASK_CHOICES_COUNT];
        let shape : MaskShape = mask_shape(mask_type, row_seed);
        let multiplier : i32 = mask_multiplier(mask_type, shape.width);

        // Use natural mask height * multiplier
        var row_height : i32 = shape.height * multiplier;
        row_heights[u32(row_index)] = row_height;
        if (row_height <= 0) { break; }

        var start_y : i32 = height - bottom_padding - row_height;
        if (start_y < 0) {
            row_height = row_height + start_y;
            start_y = 0;
        }
        if (row_height <= 0) {
            break;
        }

        if (coord_y >= start_y && coord_y < start_y + row_height) {
            let local_y : i32 = coord_y - start_y;
            // Direct mapping: row_height pixels map to shape.height * multiplier mask pixels
            let source_y : f32 = f32(local_y) * f32(shape.height) / f32(row_height);
            let mask_y0 : i32 = clamp_i32(i32(floor(source_y)), 0, shape.height - 1);
            let mask_y1 : i32 = clamp_i32(mask_y0 + 1, 0, shape.height - 1);
            let frac_y : f32 = fract(source_y);

            // Compute horizontal repeats to fill width
            // Python: width = int(shape[1] / multiplier) quantized to mask_shape[1]
            var base_width : i32 = width / max(multiplier, 1);
            if (base_width < shape.width) { base_width = shape.width; }
            // Quantize to be evenly divisible by mask width (matches Python line 3007)
            let quantized_width : i32 = shape.width * max(1, base_width / shape.width);
            var repeats : i32 = max(1, quantized_width / max(shape.width, 1));
            let glyph_width : i32 = max(shape.width, 1);
            let glyph_padding : i32 = mask_padding(mask_type, glyph_width);
            let padded_glyph_width : i32 = max(glyph_width + glyph_padding, 1);
            let row_width : i32 = padded_glyph_width * repeats;
            let row_width_f : f32 = max(f32(row_width), 1.0);
            let width_f : f32 = max(f32(width), 1.0);

            // Each row scrolls independently, with speed proportional to character width
            let row_speed_factor : f32 = random_float(row_seed, 199u) * 0.5 + 0.75; // 0.75 to 1.25
            let width_factor : f32 = f32(glyph_width) / 6.0; // Normalize to typical width of ~6
            let scroll_offset : f32 = time_value * speed_value * row_width_f * row_speed_factor * width_factor;

            let sample_x_f : f32 = (f32(coord_x) + 0.5) / width_f * row_width_f;
            // Smooth sub-pixel scrolling; positive offset scrolls content to the left.
            let scrolled_x : f32 = sample_x_f + scroll_offset;
            let wrapped_x : f32 = scrolled_x - floor(scrolled_x / row_width_f) * row_width_f;
            var sample_x : i32 = i32(floor(wrapped_x));
            let frac_x : f32 = fract(wrapped_x);
            let glyph_index : i32 = clamp_i32(sample_x / padded_glyph_width, 0, max(repeats - 1, 0));
            let local_x : i32 = sample_x % padded_glyph_width;

            let glyph_seed : u32 = combine_seed(row_seed, u32(glyph_index) * 131u + 17u);
            
            // Sample current and next x position for horizontal interpolation
            let value00 : f32 = sample_padded_mask(mask_type, local_x, mask_y0, glyph_width, shape.height, glyph_padding, glyph_seed, row_seed);
            let value01 : f32 = sample_padded_mask(
                mask_type,
                (local_x + 1) % padded_glyph_width,
                mask_y0,
                glyph_width,
                shape.height,
                glyph_padding,
                glyph_seed,
                row_seed,
            );
            let value10 : f32 = sample_padded_mask(mask_type, local_x, mask_y1, glyph_width, shape.height, glyph_padding, glyph_seed, row_seed);
            let value11 : f32 = sample_padded_mask(
                mask_type,
                (local_x + 1) % padded_glyph_width,
                mask_y1,
                glyph_width,
                shape.height,
                glyph_padding,
                glyph_seed,
                row_seed,
            );
            
            // Bilinear interpolation for smooth scrolling
            let value_y0 : f32 = lerp_f32(value00, value01, frac_x);
            let value_y1 : f32 = lerp_f32(value10, value11, frac_x);
            let row_value : f32 = lerp_f32(value_y0, value_y1, frac_y);
            mask_value = max(mask_value, row_value);
        }

        bottom_padding = bottom_padding + row_height + 2;
        if (bottom_padding >= height + row_height) {
            break;
        }

        row_index = row_index + 1;
    }

    return clamp(mask_value, 0.0, 1.0);
}

fn apply_blend(src : f32, offset_value : f32, mask_val : f32, alpha : f32) -> f32 {
    let shadow_alpha : f32 = alpha * (1.0 / 3.0);
    let first : f32 = mix(src, offset_value, shadow_alpha);
    let highlight : f32 = max(mask_val, first);
    let final_val : f32 = mix(first, highlight, alpha);
    return clamp(final_val, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.width), 1u);
    let height : u32 = max(as_u32(params.height), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let channel_count : u32 = sanitized_channel_count(params.channels);

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let src : vec4<f32> = textureLoad(input_texture, coords, 0);

    // Use a stable seed independent of time/speed so glyphs do not change each frame.
    let base_seed : u32 = combine_seed(
        hash_mix(bitcast<u32>(params.width)),
        hash_mix(bitcast<u32>(params.height)) ^ 0x9e3779b9u,
    );
    let alpha : f32 = clamp(0.5 + random_float(base_seed, 197u) * 0.25, 0.0, 1.0);

    // Flip Y so ticker rows anchor to the bottom of the final image instead of the top.
    let ticker_y : i32 = height_i - 1 - coords.y;
    let mask_val : f32 = ticker_mask(coords.x, ticker_y, width_i, height_i, base_seed, params.time, params.speed);
    let offset_mask : f32 = ticker_mask(coords.x - 1, ticker_y + 1, width_i, height_i, base_seed, params.time, params.speed);

    let offset_value_r : f32 = src.x - offset_mask;
    let offset_value_g : f32 = src.y - offset_mask;
    let offset_value_b : f32 = src.z - offset_mask;

    var result : vec4<f32> = vec4<f32>(0.0);
    result.x = apply_blend(src.x, offset_value_r, mask_val, alpha);
    result.y = apply_blend(src.y, offset_value_g, mask_val, alpha);
    result.z = apply_blend(src.z, offset_value_b, mask_val, alpha);
    result.w = src.w;

    let base_index : u32 = pixel_base_index(gid.x, gid.y, width);
    if (channel_count >= 1u) {
        output_buffer[base_index + 0u] = result.x;
    }
    if (channel_count >= 2u) {
        output_buffer[base_index + 1u] = result.y;
    }
    if (channel_count >= 3u) {
        output_buffer[base_index + 2u] = result.z;
    }
    if (channel_count >= 4u) {
        output_buffer[base_index + 3u] = result.w;
    }
    for (var c : u32 = channel_count; c < CHANNEL_COUNT; c = c + 1u) {
        output_buffer[base_index + c] = select(0.0, result.w, c == 3u);
    }
}

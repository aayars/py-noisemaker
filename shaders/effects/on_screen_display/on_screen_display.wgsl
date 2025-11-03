const INV_UINT_RANGE : f32 = 1.0 / 4294967296.0;

const BANK_OCR_ATLAS : array<array<array<f32, 7>, 8>, 10> = array<array<array<f32, 7>, 8>, 10>(
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 7>, 8>(
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 7>(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
);

const ALPHANUM_NUMERIC_ATLAS : array<array<array<f32, 6>, 6>, 10> = array<array<array<f32, 6>, 6>, 10>(
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    )
);

const ALPHANUM_HEX_ATLAS : array<array<array<f32, 6>, 6>, 16> = array<array<array<f32, 6>, 6>, 16>(
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        array<f32, 6>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(0.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    ),
    array<array<f32, 6>, 6>(
        array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        array<f32, 6>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
);

// Renders a bank OCR / hexadecimal style on-screen display overlay matching
// the CPU implementation. A narrow row of glyphs is composited in the upper
// right corner with a randomized atlas selection animated by simplex noise.

const TAU : f32 = 6.283185307179586;

struct OnScreenDisplayParams {
    size : vec4<f32>,       // (width, height, channels, unused)
    time_speed : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : OnScreenDisplayParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
    var v : vec3<u32> = v_in * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return v;
}

fn random_u32(base_seed : u32, salt : u32) -> u32 {
    let mixed : vec3<u32> = vec3<u32>(
        (base_seed) ^ ((salt * 0x9e3779b9u) + 0x632be59bu),
        ((base_seed + 0x7f4a7c15u)) ^ ((salt * 0x165667b1u) + 0x85ebca6bu),
        ((base_seed ^ 0x27d4eb2du) + (salt * 0x94d049bbu) + 0x5bf03635u)
    );
    return pcg3d(mixed).x;
}

fn random_float(base_seed : u32, salt : u32) -> f32 {
    return f32(random_u32(base_seed, salt)) * INV_UINT_RANGE;
}

fn random_range(base_seed : u32, salt : u32, min_value : i32, max_value : i32) -> i32 {
    var lo : i32 = min_value;
    var hi : i32 = max_value;
    if (lo > hi) {
        let tmp : i32 = lo;
        lo = hi;
        hi = tmp;
    }
    let span : i32 = hi - lo;
    if (span <= 0) {
        return lo;
    }
    let scaled : f32 = random_float(base_seed, salt) * f32(span + 1);
    let offset : i32 = clamp(i32(floor(scaled)), 0, span);
    return lo + offset;
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x : vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r : vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise(v : vec3<f32>) -> f32 {
    let c : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let d : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let i0 : vec3<f32> = floor(v + dot(v, vec3<f32>(c.y)));
    let x0 : vec3<f32> = v - i0 + dot(i0, vec3<f32>(c.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(c.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(c.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(d.y);

    let i = mod289_vec3(i0);
    let p = permute(
        permute(
            permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        )
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(d.w, d.y, d.z) - vec3<f32>(d.x, d.z, d.x);

    let j : vec4<f32> = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ : vec4<f32> = floor(j * ns.z);
    let y_ : vec4<f32> = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.y;
    let y = y_ * ns.x + ns.y;
    let h = 1.0 - abs(x) - abs(y);

    let b0 : vec4<f32> = vec4<f32>(x.x, x.y, y.x, y.y);
    let b1 : vec4<f32> = vec4<f32>(x.z, x.w, y.z, y.w);

    let s0 : vec4<f32> = floor(b0) * 2.0 + 1.0;
    let s1 : vec4<f32> = floor(b1) * 2.0 + 1.0;
    let sh : vec4<f32> = -step(h, vec4<f32>(0.0));

    let a0 : vec4<f32> = vec4<f32>(
        b0.x,
        b0.z,
        b0.y,
        b0.w
    ) + vec4<f32>(
        s0.x,
        s0.z,
        s0.y,
        s0.w
    ) * vec4<f32>(
        sh.x,
        sh.x,
        sh.y,
        sh.y
    );
    let a1 : vec4<f32> = vec4<f32>(
        b1.x,
        b1.z,
        b1.y,
        b1.w
    ) + vec4<f32>(
        s1.x,
        s1.z,
        s1.y,
        s1.w
    ) * vec4<f32>(
        sh.z,
        sh.z,
        sh.w,
        sh.w
    );

    let g0 : vec3<f32> = vec3<f32>(a0.x, a0.y, h.x);
    let g1 : vec3<f32> = vec3<f32>(a0.z, a0.w, h.y);
    let g2 : vec3<f32> = vec3<f32>(a1.x, a1.y, h.z);
    let g3 : vec3<f32> = vec3<f32>(a1.z, a1.w, h.w);

    let norm : vec4<f32> = taylor_inv_sqrt(vec4<f32>(
        dot(g0, g0),
        dot(g1, g1),
        dot(g2, g2),
        dot(g3, g3)
    ));

    let g0n : vec3<f32> = g0 * norm.x;
    let g1n : vec3<f32> = g1 * norm.y;
    let g2n : vec3<f32> = g2 * norm.z;
    let g3n : vec3<f32> = g3 * norm.w;

    let m0 : f32 = max(0.6 - dot(x0, x0), 0.0);
    let m1 : f32 = max(0.6 - dot(x1, x1), 0.0);
    let m2 : f32 = max(0.6 - dot(x2, x2), 0.0);
    let m3 : f32 = max(0.6 - dot(x3, x3), 0.0);

    let m0sq : f32 = m0 * m0;
    let m1sq : f32 = m1 * m1;
    let m2sq : f32 = m2 * m2;
    let m3sq : f32 = m3 * m3;

    return 42.0 * (
        m0sq * m0sq * dot(g0n, x0) +
        m1sq * m1sq * dot(g1n, x1) +
        m2sq * m2sq * dot(g2n, x2) +
        m3sq * m3sq * dot(g3n, x3)
    );
}

fn sample_atlas(mask_type : u32, glyph_index : u32, row : u32, column : u32) -> f32 {
    // Atlases are stored with row 0 at top, which matches Python's indexing
    // No flip needed here - the issue is in how we calculate inner_y
    switch mask_type {
        case 0u: {
            return BANK_OCR_ATLAS[min(glyph_index, 9u)][min(row, 7u)][min(column, 6u)];
        }
        case 1u: {
            return ALPHANUM_HEX_ATLAS[min(glyph_index, 15u)][min(row, 5u)][min(column, 5u)];
        }
        default: {
            return ALPHANUM_NUMERIC_ATLAS[min(glyph_index, 9u)][min(row, 5u)][min(column, 5u)];
        }
    }
}

fn compute_seed(width_u : u32, height_u : u32) -> u32 {
    // Seed should be constant for all frames - only time parameter should change
    // This ensures the same font and noise seed are used throughout animation
    var seed : u32 = (width_u * 0x9e3779b9u) ^ (height_u * 0x7f4a7c15u);
    seed = seed ^ 0x632be59bu;  // Additional mixing constant
    return seed;
}

fn glyph_noise_value(
    base_seed : u32,
    mask_choice : u32,
    glyph_x : i32,
    glyph_y : i32,
    time_value : f32,
    speed_value : f32
) -> f32 {
    // Python: uv_noise = simplex(uv_shape, time=time, seed=rng.random_int(1, 65536), speed=speed)
    // where uv_shape = [1, glyph_count]
    // Then simplex samples at integer coords: noise3d(x, y, z) for x in [0..glyph_count-1], y=0
    // with z = cos(2*pi*time) * speed
    //
    // Python creates an OpenSimplex instance from seed: os, _ = from_seed(base_seed)
    // Then: glyph_index = int(uv_noise[uv_y][uv_x] * len(atlas))
    
    // Match Python's simplex sampling exactly:
    // angle = math.pi * 2 * time
    // z = math.cos(angle) * speed
    // val = os.noise3d(x, y, z)
    // return (val + 1) * 0.5
    
    // Generate a consistent noise seed from base_seed (Python: rng.random_int(1, 65536))
    // Use this to offset the simplex coordinates so different seeds produce different patterns
    let noise_seed : u32 = random_u32(base_seed, 13u);
    let seed_offset : vec3<f32> = vec3<f32>(
        f32((noise_seed >> 16u) & 0xFFFFu) * 0.1,
        f32(noise_seed & 0xFFFFu) * 0.1,
        f32((noise_seed >> 8u) & 0xFFu) * 0.1
    );
    
    let angle : f32 = TAU * time_value;
    let z_coord : f32 = cos(angle) * speed_value;
    
    // Sample at integer glyph grid coordinates plus seed offset
    let sample_pos : vec3<f32> = vec3<f32>(
        f32(glyph_x),
        f32(glyph_y),
        z_coord
    ) + seed_offset;
    
    // Simplex returns [-1, 1], Python normalizes to [0, 1]
    let noise_value : f32 = simplex_noise(sample_pos) * 0.5 + 0.5;
    
    return clamp(noise_value, 0.0, 1.0);
}

fn overlay_value_at(
    coord : vec2<i32>,
    dims : vec2<i32>,
    base_seed : u32,
    time_value : f32,
    speed_value : f32
) -> f32 {
    let base_segment : i32 = i32(floor(f32(dims.x) / 24.0));
    if (base_segment <= 0) {
        return 0.0;
    }

    let mask_choice : u32 = random_u32(base_seed, 11u) % 3u;
    var mask_width : i32 = 6;
    var mask_height : i32 = 6;
    var atlas_length : u32 = 10u;
    if (mask_choice == 0u) {
        mask_width = 7;
        mask_height = 8;
        atlas_length = 10u;
    } else if (mask_choice == 1u) {
        mask_width = 6;
        mask_height = 6;
        atlas_length = 16u;
    }

    if (base_segment < mask_width) {
        return 0.0;
    }

    let scale : i32 = max(base_segment / mask_width, 1);
    let glyph_height : i32 = mask_height * scale;
    let glyph_width : i32 = mask_width * scale;
    let glyph_count : i32 = random_range(base_seed, 23u, 3, 6);
    if (glyph_count <= 0) {
        return 0.0;
    }

    let overlay_width : i32 = glyph_width * glyph_count;
    let overlay_height : i32 = glyph_height;
    if (overlay_width <= 0 || overlay_height <= 0) {
        return 0.0;
    }

    // Python padding: [[25, shape[0] - height - 25], [shape[1] - width - 25, 25], [0, 0]]
    // Padding format is [[top, bottom], [left, right], [channels_before, channels_after]]
    // Content starts at (left_padding, top_padding) = (shape[1] - width - 25, 25)
    var origin_x : i32 = dims.x - overlay_width - 25;  // Right side with 25px margin
    if (origin_x < 0) {
        origin_x = 0;
    }
    // TESTING: Try bottom-right to see if coordinate system is inverted
    var origin_y : i32 = dims.y - overlay_height - 25;  // Bottom with 25px margin
    if (origin_y < 0) {
        origin_y = 0;
    }

    if (coord.x < origin_x || coord.x >= origin_x + overlay_width) {
        return 0.0;
    }
    if (coord.y < origin_y || coord.y >= origin_y + overlay_height) {
        return 0.0;
    }

    let local_x : i32 = coord.x - origin_x;
    let local_y : i32 = coord.y - origin_y;

    let stride : i32 = glyph_width;
    if (stride <= 0) {
        return 0.0;
    }

    // Compute which glyph cell we're in (matching Python's freq-based grid)
    let glyph_x : i32 = local_x / glyph_width;
    let glyph_y : i32 = local_y / glyph_height;
    
    // Ensure glyph indices are valid
    if (glyph_x < 0 || glyph_x >= glyph_count || glyph_y < 0 || glyph_y >= 1) {
        return 0.0;
    }

    // Sample noise at this glyph cell's position to determine which glyph to show
    // Python: glyph_index = int(uv_noise[uv_y][uv_x] * len(atlas))
    let glyph_noise : f32 = glyph_noise_value(base_seed, mask_choice, glyph_x, glyph_y, time_value, speed_value);
    let glyph_index : u32 = min(u32(floor(glyph_noise * f32(atlas_length))), atlas_length - 1u);

    // Now determine position within the glyph
    // Python: atlas[glyph_index][y % shape[0]][x % shape[1]]
    // where y and x are absolute pixel positions within the glyph cell
    let inner_x : i32 = (local_x % glyph_width) / scale;
    
    // For Y: need to flip because atlas row 0 is top of glyph, but we may be rendering bottom-up
    // Actually, let's check: if local_y increases downward within the cell,
    // and atlas row 0 is the top, then inner_y = local_y is correct
    // But if glyphs appear flipped, we need to invert within the glyph
    let inner_y_raw : i32 = (local_y % glyph_height) / scale;
    let inner_y : i32 = (mask_height - 1) - inner_y_raw;  // Flip vertically within glyph
    
    // Clamp to mask dimensions
    let clamped_inner_x : i32 = clamp(inner_x, 0, mask_width - 1);
    let clamped_inner_y : i32 = clamp(inner_y, 0, mask_height - 1);

    return sample_atlas(mask_choice, glyph_index, u32(clamped_inner_y), u32(clamped_inner_x));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = as_u32(params.size.x);
    let height_u : u32 = as_u32(params.size.y);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let dims : vec2<i32> = vec2<i32>(i32(width_u), i32(height_u));
    let base_seed : u32 = compute_seed(width_u, height_u);
    let overlay_value : f32 = clamp01(overlay_value_at(coord, dims, base_seed, params.time_speed.x, params.time_speed.y));
    let alpha : f32 = mix(0.5, 0.75, clamp01(random_float(base_seed, 7u)));

    let sample : vec4<f32> = textureLoad(input_texture, coord, 0);
    let pixel_index : u32 = gid.y * width_u + gid.x;
    let base_index : u32 = pixel_index * 4u;

    let base_r : f32 = clamp01(sample.x);
    let base_g : f32 = clamp01(sample.y);
    let base_b : f32 = clamp01(sample.z);
    let base_a : f32 = clamp01(sample.w);

    let highlight_r : f32 = max(base_r, overlay_value);
    let highlight_g : f32 = max(base_g, overlay_value);
    let highlight_b : f32 = max(base_b, overlay_value);

    let final_r : f32 = clamp01(mix(base_r, highlight_r, alpha));
    let final_g : f32 = clamp01(mix(base_g, highlight_g, alpha));
    let final_b : f32 = clamp01(mix(base_b, highlight_b, alpha));

    output_buffer[base_index + 0u] = final_r;
    output_buffer[base_index + 1u] = final_g;
    output_buffer[base_index + 2u] = final_b;
    output_buffer[base_index + 3u] = base_a;
}

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

fn compute_seed(width_u : u32, height_u : u32, time_value : f32, speed_value : f32) -> u32 {
    let time_bits : u32 = bitcast<u32>(time_value);
    let speed_bits : u32 = bitcast<u32>(speed_value);
    var seed : u32 = (width_u * 0x9e3779b9u) ^ (height_u * 0x7f4a7c15u);
    seed = seed ^ ((time_bits * 0x632be59bu) + 0x94d049bbu);
    seed = seed ^ ((speed_bits * 0x165667b1u) + 0x27d4eb2du);
    return seed;
}

fn glyph_noise_value(
    base_seed : u32,
    mask_choice : u32,
    uv_x_index : i32,
    uv_y_index : i32,
    time_value : f32,
    speed_value : f32
) -> f32 {
    let base_salt : u32 = 401u + mask_choice * 97u;
    let jitter_bits : u32 = random_u32(base_seed, base_salt);
    let jitter_x : f32 = f32(jitter_bits & 0x3ffu) * (1.0 / 1024.0);
    let jitter_y : f32 = f32((jitter_bits >> 10) & 0x3ffu) * (1.0 / 1024.0);
    let jitter_z : f32 = f32((jitter_bits >> 20) & 0x3ffu) * (1.0 / 1024.0);

    let seed_bits : u32 = random_u32(base_seed, base_salt ^ 0x9e3779b9u);
    let offset_x : f32 = f32(seed_bits & 0x3ffu) * (1.0 / 64.0);
    let offset_y : f32 = f32((seed_bits >> 10) & 0x3ffu) * (1.0 / 64.0);
    let offset_z : f32 = f32((seed_bits >> 20) & 0x3ffu) * (1.0 / 4096.0);

    let angle : f32 = time_value * TAU;
    let z : f32 = cos(angle) * speed_value + jitter_z + offset_z;

    let sample : vec3<f32> = vec3<f32>(
        f32(uv_x_index) + jitter_x + offset_x + f32(mask_choice) * 5.0,
        f32(uv_y_index) + jitter_y + offset_y + f32(mask_choice) * 3.0,
        z
    );

    return clamp(simplex_noise(sample) * 0.5 + 0.5, 0.0, 1.0);
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

    var origin_x : i32 = dims.x - overlay_width - 25;
    if (origin_x < 0) {
        origin_x = 0;
    }
    var origin_y : i32 = 25;
    if (origin_y + overlay_height > dims.y) {
        origin_y = max(dims.y - overlay_height, 0);
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

    let safe_mask_width : i32 = max(mask_width, 1);
    let safe_mask_height : i32 = max(mask_height, 1);
    let uv_columns : i32 = max(scale * glyph_count, 1);
    let uv_rows : i32 = max(scale, 1);
    let uv_x_index : i32 = clamp(local_x / safe_mask_width, 0, uv_columns - 1);
    let uv_y_index : i32 = clamp(local_y / safe_mask_height, 0, uv_rows - 1);

    let glyph_noise : f32 = glyph_noise_value(base_seed, mask_choice, uv_x_index, uv_y_index, time_value, speed_value);
    let glyph_index : u32 = min(u32(floor(glyph_noise * f32(atlas_length))), atlas_length - 1u);

    let inner_x : i32 = clamp((local_x % stride) / max(scale, 1), 0, mask_width - 1);
    let inner_y : i32 = clamp(local_y / max(scale, 1), 0, mask_height - 1);

    return sample_atlas(mask_choice, glyph_index, u32(inner_y), u32(inner_x));
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
    let base_seed : u32 = compute_seed(width_u, height_u, params.time_speed.x, params.time_speed.y);
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

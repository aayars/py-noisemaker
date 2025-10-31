FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libhdf5-dev \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy only files needed for installation
COPY pyproject.toml README.md LICENSE ./
COPY noisemaker ./noisemaker
COPY dsl ./dsl

# Build and install the package
RUN pip install --no-cache-dir .


FROM python:3.13-slim

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-310 \
        ffmpeg && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/noisemaker /usr/local/bin/noisemaker
COPY --from=builder /usr/local/bin/magic-mashup /usr/local/bin/magic-mashup
COPY --from=builder /usr/local/bin/mood /usr/local/bin/mood

# Create non-root user
RUN useradd -m -u 1000 noisemaker && \
    mkdir -p /output && \
    chown -R noisemaker:noisemaker /output

USER noisemaker
WORKDIR /output

# Verify installation
RUN noisemaker --help > /dev/null

CMD ["noisemaker", "--help"]

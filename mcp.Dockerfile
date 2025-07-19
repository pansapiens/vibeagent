# Use a lightweight Python base image
FROM python:3.12-slim

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Node.js and other utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    git \
    file \
    jq \
    wget \
    vim \
    nano \
    procps \
    rsync \
    tar \
    zip \
    unzip \
    grep \
    sed \
    gawk \
    netcat-openbsd \
    iputils-ping \
    dnsutils \
    lsb-release \
    build-essential \
    pkg-config \
    golang \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS version) from NodeSource
# See: https://github.com/nodesource/distributions
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Install uv using pip
RUN pip install --no-cache-dir uv

# Create a non-root user 'agent'
# The UID/GID will be overridden at runtime with the host user's UID/GID,
# but a user and home directory must exist.
RUN useradd -m -u 1000 -s /bin/bash agent

# The workdir will be set by the calling script via --workdir
# but we set it here for clarity and for anyone using the image directly.
WORKDIR /home/agent

# Switch to the non-root user
USER agent

# Set a default command to something harmless that provides a shell.
CMD ["/bin/bash"] 
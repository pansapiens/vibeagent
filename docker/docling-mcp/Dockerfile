FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Clone the docling-mcp repository
RUN git clone https://github.com/docling-project/docling-mcp.git .

# Install dependencies using uv
RUN pip install --no-cache-dir .

# Set the default command
CMD ["docling-mcp-server", "--transport", "stdio"]

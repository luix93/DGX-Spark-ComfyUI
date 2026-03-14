# ComfyUIMini - Mobile-friendly UI for ComfyUI
# Lightweight Node.js container that proxies to ComfyUI
FROM node:20-alpine

LABEL maintainer="ComfyUI"
LABEL description="ComfyUIMini mobile UI for ComfyUI on DGX Spark"

WORKDIR /app

# Install git for cloning, then clean up
RUN apk add --no-cache git

# Clone ComfyUIMini
ARG COMFYUIMINI_REF=main
RUN git clone https://github.com/ImDarkTom/ComfyUIMini.git . && \
    git checkout ${COMFYUIMINI_REF}

# Install all deps, build TypeScript, then prune dev dependencies
RUN npm ci && \
    npm run build && \
    npm prune --omit=dev

# Create config directory and default config
# Config is overridden at runtime via NODE_CONFIG env var
RUN mkdir -p config && \
    cp config/default.example.json config/default.json

# Create directories for workflows and ensure proper permissions
RUN mkdir -p workflows && \
    chown -R node:node /app

# Switch to non-root user
USER node

EXPOSE 3000

# Health check - verify the server responds
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Project uses tsx to run TypeScript directly (not compiled to dist/)
CMD ["npm", "start"]

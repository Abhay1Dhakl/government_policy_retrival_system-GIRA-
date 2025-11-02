# Use the official Node.js 18 image as base
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps

# Install required packages for canvas and node-gyp
RUN apk add --no-cache \
    libc6-compat \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    musl-dev \
    giflib-dev \
    pixman-dev \
    pangomm-dev \
    libjpeg-turbo-dev \
    freetype-dev

WORKDIR /app

# Install dependencies based on the preferred package manager
COPY package.json package-lock.json* ./

# Set Python path for node-gyp
ENV PYTHON=/usr/bin/python3

# Install all dependencies (including dev dependencies for npm run dev)
RUN npm ci

# Development image
FROM base AS runner
WORKDIR /app

# Install runtime dependencies for canvas
RUN apk add --no-cache \
    cairo \
    jpeg \
    pango \
    musl \
    giflib \
    pixman \
    pangomm \
    libjpeg-turbo \
    freetype \
    python3

ENV NODE_ENV development
ENV NEXT_TELEMETRY_DISABLED 1

# Copy node_modules and source code
COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

# Use npm run dev for development
CMD ["npm", "run", "dev"]

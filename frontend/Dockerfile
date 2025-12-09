# # Multi-stage build for Vite React app

# # Stage 1: Build
# FROM node:20-alpine AS builder
# WORKDIR /app

# # Declare build arguments (Vite needs these during build)
# ARG VITE_SUPABASE_URL
# ARG VITE_SUPABASE_ANON_KEY
# ARG VITE_API_URL

# # Copy package files
# COPY package.json package-lock.json ./

# # Install dependencies
# RUN npm ci

# # Copy source code
# COPY . .

# # Build the application (Vite reads ARGs as environment variables)
# RUN npm run build

# # Stage 2: Production with Nginx
# FROM nginx:alpine
# COPY --from=builder /app/dist /usr/share/nginx/html
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# EXPOSE 80

# HEALTHCHECK --interval=30s --timeout=3s \
#   CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# CMD ["nginx", "-g", "daemon off;"]


# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
RUN npm run build

# Stage 2: Production
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
# Use standalone config by default, can override in docker-compose
COPY nginx.standalone.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
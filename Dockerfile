# Dockerfile for Crossword Generator
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install Python and pip for the Python generator
RUN apk add --no-cache python3 py3-pip wget

# Copy package files first for better caching
COPY package*.json ./

# Install Node.js dependencies
RUN npm install

# Copy application files
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start the server
CMD ["node", "server.js"]

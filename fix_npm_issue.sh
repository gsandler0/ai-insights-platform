#!/bin/bash

echo "Fixing npm build issue..."

# Stop any running containers
docker-compose down

# Fix the web app Dockerfile
cat > app/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (use npm install instead of ci)
RUN npm install --only=production

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Create logs directory
RUN mkdir -p /app/logs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start the application
CMD ["npm", "start"]
EOF

# Generate package-lock.json by running npm install locally
cd app
echo "Generating package-lock.json..."
npm install --package-lock-only
cd ..

echo "Rebuild and start the platform..."
docker-compose up --build -d

echo ""
echo "✅ Fixed! The platform should now start successfully."
echo "🌐 Access it at: http://localhost"
echo ""
echo "Monitor the startup with:"
echo "  ./scripts/logs.sh"
echo ""
echo "Check individual services:"
echo "  ./scripts/logs.sh web-app"
echo "  ./scripts/logs.sh ollama"
echo "  ./scripts/logs.sh mcp-server"

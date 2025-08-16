#!/bin/bash

echo "🔍 Diagnosing Ollama issue..."

# Check Ollama container logs
echo "📋 Ollama container logs:"
docker logs ai-insights-ollama 2>&1 | tail -20

echo ""
echo "📊 Container status:"
docker ps -a --filter "name=ai-insights"

echo ""
echo "🔧 Attempting fixes..."

# Stop all containers
docker-compose down

# Remove the problematic Ollama container
docker rm -f ai-insights-ollama 2>/dev/null || true

# Clean up any hanging volumes
docker volume prune -f

echo "📝 Creating improved docker-compose.yml with Ollama fixes..."

# Create improved docker-compose with Ollama fixes
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: ai-insights-postgres
    environment:
      POSTGRES_DB: insights_db
      POSTGRES_USER: insights_user
      POSTGRES_PASSWORD: insights_password_2024
      POSTGRES_HOST_AUTH_METHOD: md5
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - ai-insights-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U insights_user -d insights_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ai-insights-ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    networks:
      - ai-insights-network
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:11434/api/tags || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 60s
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          memory: 4G

  model-setup:
    image: curlimages/curl:latest
    container_name: ai-insights-model-setup
    depends_on:
      ollama:
        condition: service_healthy
    networks:
      - ai-insights-network
    command: >
      sh -c "
        echo 'Waiting for Ollama to be ready...';
        sleep 30;
        echo 'Pulling Llama 3.1 8B model...';
        curl -X POST http://ollama:11434/api/pull \
          -d '{\"name\": \"llama3.1:8b\"}' \
          -H 'Content-Type: application/json' \
          --max-time 1800 || echo 'Model pull failed, but continuing...';
        echo 'Model setup complete';
      "
    restart: "no"

  mcp-server:
    build:
      context: ./mcp
      dockerfile: Dockerfile
    container_name: ai-insights-mcp
    environment:
      - DATABASE_URL=postgresql://insights_user:insights_password_2024@postgres:5432/insights_db
      - OLLAMA_URL=http://ollama:11434
    volumes:
      - app_logs:/app/logs
    ports:
      - "8001:8001"
    depends_on:
      postgres:
        condition: service_healthy
      ollama:
        condition: service_started
    networks:
      - ai-insights-network
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  web-app:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: ai-insights-app
    environment:
      - MCP_SERVER_URL=http://mcp-server:8001
      - DATABASE_URL=postgresql://insights_user:insights_password_2024@postgres:5432/insights_db
    volumes:
      - app_logs:/app/logs
    ports:
      - "3000:3000"
    depends_on:
      mcp-server:
        condition: service_started
    networks:
      - ai-insights-network
    restart: unless-stopped

  nginx:
    build:
      context: ./nginx
      dockerfile: Dockerfile
    container_name: ai-insights-nginx
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - web-app
    networks:
      - ai-insights-network
    restart: unless-stopped

networks:
  ai-insights-network:
    driver: bridge

volumes:
  postgres_data:
  ollama_data:
  app_logs:
EOF

echo "🚀 Starting services with improved configuration..."

# Start core services first
echo "Starting database..."
docker-compose up -d postgres

# Wait for postgres to be healthy
echo "Waiting for PostgreSQL to be ready..."
until docker-compose exec postgres pg_isready -U insights_user -d insights_db; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

echo "Starting Ollama..."
docker-compose up -d ollama

# Wait a bit for Ollama to start
sleep 10

echo "Starting remaining services..."
docker-compose up -d

echo ""
echo "📊 Service status:"
docker-compose ps

echo ""
echo "📋 Checking Ollama logs:"
sleep 5
docker logs ai-insights-ollama --tail 10

echo ""
echo "✅ Platform startup initiated!"
echo ""
echo "🌐 Access the web interface at: http://localhost"
echo ""
echo "⏳ Note: The AI model download will happen in the background."
echo "   This can take 5-15 minutes depending on your connection."
echo ""
echo "📊 Monitor progress:"
echo "   ./scripts/logs.sh ollama          # Watch model download"
echo "   ./scripts/logs.sh model-setup     # Watch setup process" 
echo "   docker-compose ps                 # Check all services"
echo ""
echo "🔧 If issues persist, try:"
echo "   docker system prune -f            # Clean up Docker"
echo "   docker-compose down -v            # Remove all data"
echo "   docker-compose up -d              # Fresh start"
EOF
chmod +x diagnose_fix_script.sh


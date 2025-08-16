#!/bin/bash

echo "🔍 Checking Ollama and model status..."

echo "📊 Current service status:"
docker-compose ps

echo ""
echo "🤖 Checking if Ollama is responding:"
curl -s http://localhost:11434/api/tags

echo ""
echo "📋 Model setup container logs:"
docker logs ai-insights-model-setup --tail 20

echo ""
echo "📋 Ollama container logs (last 10 lines):"
docker logs ai-insights-ollama --tail 10

echo ""
echo "🔧 Let's manually trigger model download..."

# Check if model-setup container is still running
MODEL_SETUP_STATUS=$(docker ps -q -f name=ai-insights-model-setup)

if [ -z "$MODEL_SETUP_STATUS" ]; then
    echo "⚠️  Model setup container not running. Starting manual download..."
    
    # Manual model download
    echo "📥 Downloading Llama 3.1 8B model manually..."
    curl -X POST http://localhost:11434/api/pull \
        -H "Content-Type: application/json" \
        -d '{"name": "llama3.1:8b"}' \
        --max-time 1800 &
    
    DOWNLOAD_PID=$!
    
    echo "🔄 Model download started (PID: $DOWNLOAD_PID)"
    echo "⏳ This will take 5-15 minutes depending on your connection..."
    
    # Monitor download progress
    echo "📊 Monitoring download progress..."
    while kill -0 $DOWNLOAD_PID 2>/dev/null; do
        echo "⏳ Still downloading... ($(date))"
        sleep 30
    done
    
    wait $DOWNLOAD_PID
    DOWNLOAD_RESULT=$?
    
    if [ $DOWNLOAD_RESULT -eq 0 ]; then
        echo "✅ Model download completed!"
    else
        echo "❌ Model download failed or timed out"
    fi
else
    echo "✅ Model setup container is still running"
fi

echo ""
echo "🔍 Checking available models:"
curl -s http://localhost:11434/api/tags | jq . 2>/dev/null || curl -s http://localhost:11434/api/tags

echo ""
echo "🧪 Testing model with simple query:"
curl -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.1:8b",
        "prompt": "Hello, respond with just: Model is working!",
        "stream": false
    }' | jq -r '.response' 2>/dev/null || echo "❌ Model not ready yet"

echo ""
echo "🔄 Restarting MCP server to retry connections..."
docker-compose restart mcp-server

echo ""
echo "⏳ Waiting for MCP server to restart..."
sleep 10

echo ""
echo "🧪 Testing the web interface again:"
echo "🌐 Go to: http://localhost"
echo "🔍 Try asking: 'What are our total sales by month?'"

echo ""
echo "📊 Final service status:"
docker-compose ps

echo ""
echo "📋 Troubleshooting commands:"
echo "   # Check if model is downloaded:"
echo "   curl http://localhost:11434/api/tags"
echo ""
echo "   # Test model directly:"
echo "   curl -X POST http://localhost:11434/api/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"llama3.1:8b\", \"prompt\": \"Hello\", \"stream\": false}'"
echo ""
echo "   # Check MCP server logs:"
echo "   docker logs ai-insights-mcp --tail 20"
echo ""
echo "   # Restart everything if needed:"
echo "   docker-compose restart"

# Alternative: Download a smaller model if the main one fails
echo ""
echo "🔄 If the model is taking too long, we can try a smaller model:"
echo "   # Download smaller model (much faster):"
echo "   curl -X POST http://localhost:11434/api/pull -d '{\"name\": \"llama3.2:1b\"}'"
echo ""
echo "   # Then update MCP server to use it (edit mcp/server.py and change model name)"

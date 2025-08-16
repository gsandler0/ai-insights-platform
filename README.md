# AI Insights Platform POC

A secure, scalable platform for running AI-powered analytics on sensitive business data using the Model Context Protocol (MCP).

## Architecture

- **PostgreSQL**: Secure database for your sensitive data
- **Ollama**: Local AI model server (Llama 3.1 8B)
- **MCP Server**: Model Context Protocol implementation
- **Web App**: React-based frontend for natural language queries
- **Nginx**: Reverse proxy with security headers and rate limiting

## Features

- 🔒 **Secure**: All data stays within your Docker environment
- 🤖 **AI-Powered**: Natural language to SQL query generation
- 📊 **Interactive**: Web-based interface for data exploration
- 🏗️ **Scalable**: Containerized architecture
- 🔄 **MCP Compliant**: Uses Model Context Protocol standards

## Quick Start

1. **Setup the platform:**
   ```bash
   ./scripts/start.sh
   ```

2. **Access the application:**
   - Open http://localhost in your browser
   - Wait for the AI model to download (first run only)

3. **Try sample queries:**
   - "What are our total sales by month?"
   - "Which customers have the highest revenue?"
   - "Show sales performance by region"

## Sample Data

The platform comes with realistic sales data including:
- Sales transactions with customer and product information
- Customer profiles with company and industry data
- Product catalog with pricing and margin information

## Usage Examples

### Natural Language Queries
```
"Show me the top 5 customers by total revenue"
"What are our monthly sales trends for 2024?"
"Which products have the highest profit margins?"
"How many sales did each sales rep make this quarter?"
```

### Database Schema
The platform includes these tables:
- `sales_data` - Transaction records
- `customers` - Customer information
- `products` - Product catalog
- `sales_summary` - Analytics view

## Management Commands

```bash
# Start the platform
./scripts/start.sh

# Stop the platform  
./scripts/stop.sh

# Restart services
./scripts/restart.sh

# View logs
./scripts/logs.sh [service_name]

# Cleanup (removes all data)
./

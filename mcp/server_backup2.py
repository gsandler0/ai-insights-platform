import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import psycopg2
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pythonjsonlogger import jsonlogger

# Configure logging
log_formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://insights_user:insights_password_2024@postgres:5432/insights_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Initialize MCP Server
server = Server("ai-insights-mcp")

# Database connection
try:
    engine = create_engine(DATABASE_URL)
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise

# Pydantic models (keeping for internal use)
class QueryRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    sql_query: str
    results: List[Dict[str, Any]]
    explanation: str
    metadata: Dict[str, Any]

# Database schema cache
_schema_cache = None
_schema_cache_time = None
SCHEMA_CACHE_TTL = 3600  # 1 hour

async def get_database_schema() -> Dict[str, Any]:
    """Get database schema with caching - UNCHANGED"""
    global _schema_cache, _schema_cache_time

    current_time = datetime.now().timestamp()
    if _schema_cache and _schema_cache_time and (current_time - _schema_cache_time) < SCHEMA_CACHE_TTL:
        return _schema_cache

    try:
        with engine.connect() as conn:
            # Get table information
            tables_query = text("""
                SELECT
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name NOT LIKE 'pg_%'
                ORDER BY table_name, ordinal_position
            """)

            result = conn.execute(tables_query)
            columns_data = result.fetchall()

            # Organize by table
            tables = {}
            for row in columns_data:
                table_name = row[0]
                if table_name not in tables:
                    tables[table_name] = {
                        "name": table_name,
                        "columns": []
                    }

                tables[table_name]["columns"].append({
                    "name": row[1],
                    "type": row[2],
                    "nullable": row[3] == "YES",
                    "default": row[4]
                })

            # Get sample data for each table
            for table_name in tables:
                try:
                    sample_query = text(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_result = conn.execute(sample_query)
                    sample_data = [dict(row._mapping) for row in sample_result]
                    tables[table_name]["sample_data"] = sample_data
                except Exception as e:
                    logger.warning(f"Could not get sample data for {table_name}: {e}")
                    tables[table_name]["sample_data"] = []

            schema = {
                "tables": list(tables.values()),
                "relationships": [
                    {
                        "from_table": "sales_data",
                        "from_column": "customer_id",
                        "to_table": "customers",
                        "to_column": "customer_id"
                    },
                    {
                        "from_table": "sales_data",
                        "from_column": "product_name",
                        "to_table": "products",
                        "to_column": "product_name"
                    }
                ]
            }

            _schema_cache = schema
            _schema_cache_time = current_time
            return schema

    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        raise RuntimeError(f"Database schema error: {str(e)}")

async def check_ollama_health() -> bool:
    """Check if Ollama is running and accessible - UNCHANGED"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            # Also verify we have at least one model
            data = response.json()
            models = data.get("models", [])
            return len(models) > 0
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return False

def create_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """Create a PostgreSQL CREATE TABLE schema string for the prompt - UNCHANGED"""
    schema_sql = ""
    
    for table in schema["tables"]:
        schema_sql += f"CREATE TABLE {table['name']} (\n"
        columns = []
        for col in table["columns"]:
            column_def = f"    {col['name']} {col['type'].upper()}"
            if not col["nullable"]:
                column_def += " NOT NULL"
            if col["default"]:
                column_def += f" DEFAULT {col['default']}"
            columns.append(column_def)
        
        schema_sql += ",\n".join(columns)
        schema_sql += "\n);\n\n"
    
    # Add relationships as comments
    if schema.get("relationships"):
        schema_sql += "-- Foreign Key Relationships:\n"
        for rel in schema["relationships"]:
            schema_sql += f"-- {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
    
    return schema_sql

async def generate_sql_query(question: str, schema: Dict[str, Any]) -> str:
    """Generate SQL query using Ollama with improved prompting - UNCHANGED"""
    # Check if Ollama is available
    if not await check_ollama_health():
        logger.error("Ollama service is not available")
        raise RuntimeError("AI service unavailable. Please ensure Ollama is running.")

    # Create PostgreSQL schema for the prompt
    schema_sql = create_schema_for_prompt(schema)
    
    # Enhanced prompt with examples for better SQL generation
    prompt = f"""### Instructions:
You are a SQL expert. Convert the question into a proper PostgreSQL query.

### Rules:
- Return ONLY the SQL query, no explanations
- Use proper JOINs when data is in multiple tables
- Use SUM(), COUNT(), GROUP BY for aggregations
- Always add LIMIT 5 at the end
- Use table aliases (c for customers, s for sales_data, etc.)

### Database Schema:
{schema_sql}

### Examples:
Question: "What are the total sales by customer?"
Answer: SELECT c.customer_name, SUM(s.amount) as total_sales FROM customers c JOIN sales_data s ON c.customer_id = s.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY total_sales DESC LIMIT 5;

Question: "Show me customers"
Answer: SELECT * FROM customers LIMIT 5;

Question: "Which products sell the most?"
Answer: SELECT p.product_name, SUM(s.quantity) as total_sold FROM products p JOIN sales_data s ON p.product_name = s.product_name GROUP BY p.product_name ORDER BY total_sold DESC LIMIT 5;

### Your Task:
Question: "{question}"
Answer: """

    try:
        logger.info(f"Generating SQL query for: {question}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get available models and prefer SQL-specific ones
            models_response = await client.get(f"{OLLAMA_URL}/api/tags")
            models_data = models_response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            logger.info(f"Available models: {available_models}")
            
            # Prefer SQL-specific models, fallback to general models
            preferred_models = ["sqlcoder", "defog-sqlcoder", "codellama", "llama3.1:8b", "llama3", "llama2"]
            model_name = None
            
            for preferred in preferred_models:
                for available in available_models:
                    if preferred in available.lower():
                        model_name = available
                        break
                if model_name:
                    break
            
            if not model_name:
                model_name = available_models[0] if available_models else "llama3.1:8b"
            
            logger.info(f"Using model: {model_name}")
            
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 200,
                    "stop": ["###", "Explanation:", "Note:", "\n\n"]
                }
            }
            
            logger.info(f"Sending request to Ollama")
            
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Ollama request failed: {response.status_code} - {error_text}")
                raise RuntimeError(f"AI service error: {response.status_code}")
            
            result = response.json()
            
            if "response" not in result:
                logger.error(f"Unexpected Ollama response format: {result}")
                raise RuntimeError("Invalid AI service response format")
            
            sql_query = result["response"].strip()
            
            if not sql_query:
                logger.error("Empty response from Ollama")
                return generate_fallback_query(question, schema)

            # Clean up the response
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Remove any extra explanatory text after the query
            lines = sql_query.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(('--', '#', '/*')):
                    continue
                if any(word in line.lower() for word in ['explanation:', 'note:', 'based on', 'this query']):
                    break
                sql_lines.append(line)
            
            sql_query = '\n'.join(sql_lines).strip()
            
            # Ensure it ends with semicolon
            if sql_query and not sql_query.endswith(";"):
                sql_query += ";"

            # Validate the query has basic SQL structure
            if not sql_query.upper().startswith('SELECT'):
                logger.warning(f"Generated query doesn't start with SELECT: {sql_query}")
                return generate_fallback_query(question, schema)

            # Ensure LIMIT is present
            if 'LIMIT' not in sql_query.upper():
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1] + ' LIMIT 5;'
                else:
                    sql_query += ' LIMIT 5'

            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return generate_fallback_query(question, schema)
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Ollama: {e}")
        return generate_fallback_query(question, schema)
    except Exception as e:
        logger.error(f"Unexpected error generating SQL query: {e}")
        return generate_fallback_query(question, schema)

def generate_fallback_query(question: str, schema: Dict[str, Any]) -> str:
    """Generate a simple fallback query when AI fails - UNCHANGED"""
    logger.info("Using fallback query generation")
    
    question_lower = question.lower()
    
    # Find the most relevant table
    table_scores = {}
    for table in schema["tables"]:
        score = 0
        table_name = table["name"].lower()
        
        # Score based on table name relevance
        if any(keyword in table_name for keyword in ["customer", "client", "user"]):
            if any(keyword in question_lower for keyword in ["customer", "client", "user"]):
                score += 10
        
        if any(keyword in table_name for keyword in ["sale", "order", "transaction"]):
            if any(keyword in question_lower for keyword in ["sale", "order", "buy", "purchase", "revenue"]):
                score += 10
                
        if any(keyword in table_name for keyword in ["product", "item", "inventory"]):
            if any(keyword in question_lower for keyword in ["product", "item", "inventory"]):
                score += 10
        
        table_scores[table["name"]] = score
    
    # Get the highest scoring table, or first table if no match
    best_table = max(table_scores.items(), key=lambda x: x[1])[0] if table_scores else schema["tables"][0]["name"]
    
    # Generate simple SELECT query
    return f"SELECT * FROM {best_table} LIMIT 5;"

async def execute_sql_query(sql_query: str) -> List[Dict[str, Any]]:
    """Execute SQL query safely - UNCHANGED except HTTPException -> RuntimeError"""
    # Basic SQL injection protection
    sql_lower = sql_query.lower().strip()

    # Only allow SELECT statements
    if not sql_lower.startswith("select"):
        raise RuntimeError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous_keywords = [
        "insert", "update", "delete", "drop", "create", "alter",
        "truncate", "exec", "execute", "sp_", "xp_", "--", "/*", "*/"
    ]

    for keyword in dangerous_keywords:
        if keyword in sql_lower:
            raise RuntimeError(f"Query contains prohibited keyword: {keyword}")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))

            # Convert to list of dictionaries
            rows = []
            for row in result:
                row_dict = {}
                for i, column in enumerate(result.keys()):
                    value = row[i]
                    # Handle special types
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    elif hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool, type(None))):
                        value = str(value)
                    row_dict[column] = value
                rows.append(row_dict)

            return rows

    except SQLAlchemyError as e:
        logger.error(f"Database error executing query: {e}")
        raise RuntimeError(f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise RuntimeError(f"Query execution error: {str(e)}")

async def generate_explanation(question: str, sql_query: str, results: List[Dict[str, Any]]) -> str:
    """Generate explanation of the query and results with fallback - UNCHANGED"""
    results_summary = f"Query returned {len(results)} rows."
    if results and len(results) > 0:
        columns = list(results[0].keys())
        results_summary += f" Columns: {', '.join(columns[:3])}"
        if len(columns) > 3:
            results_summary += f" and {len(columns)-3} more."
    
    return f"The SQL query successfully retrieved data answering '{question}'. {results_summary}"

# MCP SERVER IMPLEMENTATION

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available database resources"""
    try:
        schema = await get_database_schema()
        resources = []
        
        # Add each table as a resource
        for table in schema["tables"]:
            resources.append(
                types.Resource(
                    uri=f"db://table/{table['name']}",
                    name=f"Table: {table['name']}",
                    description=f"Database table with {len(table['columns'])} columns",
                    mimeType="application/json"
                )
            )
        
        # Add schema overview as a resource
        resources.append(
            types.Resource(
                uri="db://schema",
                name="Database Schema",
                description="Complete database schema with table definitions and relationships",
                mimeType="application/json"
            )
        )
        
        return resources
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific database resource"""
    try:
        schema = await get_database_schema()
        
        if uri == "db://schema":
            return json.dumps(schema, indent=2)
        
        if uri.startswith("db://table/"):
            table_name = uri.replace("db://table/", "")
            
            # Find the table in schema
            table_info = None
            for table in schema["tables"]:
                if table["name"] == table_name:
                    table_info = table
                    break
            
            if not table_info:
                raise RuntimeError(f"Table {table_name} not found")
            
            return json.dumps(table_info, indent=2)
        
        raise RuntimeError(f"Unknown resource URI: {uri}")
    
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        raise RuntimeError(f"Error reading resource: {str(e)}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="query_database",
            description="Execute a natural language query against the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question to convert to SQL and execute"
                    }
                },
                "required": ["question"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    if name != "query_database":
        raise RuntimeError(f"Unknown tool: {name}")
    
    if "question" not in arguments:
        raise RuntimeError("Missing required argument: question")
    
    question = arguments["question"]
    
    try:
        logger.info(f"Processing query: {question}")

        # Get database schema
        schema = await get_database_schema()

        # Generate SQL query
        sql_query = await generate_sql_query(question, schema)
        logger.info(f"Generated SQL: {sql_query}")

        # Execute query
        results = await execute_sql_query(sql_query)
        logger.info(f"Query returned {len(results)} rows")

        # Generate explanation
        explanation = await generate_explanation(question, sql_query, results)

        # Format response for MCP
        response_text = f"""SQL Query: {sql_query}

Results ({len(results)} rows):
{json.dumps(results, indent=2)}

Explanation: {explanation}"""

        return [
            types.TextContent(
                type="text",
                text=response_text
            )
        ]

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return [
            types.TextContent(
                type="text", 
                text=f"Error processing query '{question}': {str(e)}"
            )
        ]

async def main():
    """Main function to run the MCP server"""
    # Run the server using stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ai-insights-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
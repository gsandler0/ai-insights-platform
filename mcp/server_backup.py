import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Initialize FastAPI app
app = FastAPI(
    title="AI Insights MCP Server",
    description="Model Context Protocol server for AI-powered data insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
try:
    engine = create_engine(DATABASE_URL)
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    sql_query: str
    results: List[Dict[str, Any]]
    explanation: str
    metadata: Dict[str, Any]

class SchemaInfo(BaseModel):
    tables: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

# Database schema cache
_schema_cache = None
_schema_cache_time = None
SCHEMA_CACHE_TTL = 3600  # 1 hour

async def get_database_schema() -> Dict[str, Any]:
    """Get database schema with caching"""
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
        raise HTTPException(status_code=500, detail=f"Database schema error: {str(e)}")

async def check_ollama_health() -> bool:
    """Check if Ollama is running and accessible"""
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
    """Create a PostgreSQL CREATE TABLE schema string for the prompt"""
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
    """Generate SQL query using Ollama with improved prompting"""

    # Check if Ollama is available
    if not await check_ollama_health():
        logger.error("Ollama service is not available")
        raise HTTPException(status_code=503, detail="AI service unavailable. Please ensure Ollama is running.")

    # Create PostgreSQL schema for the prompt
    schema_sql = create_schema_for_prompt(schema)
    
    # Enhanced prompt specifically for SQL generation
    prompt = f"""### Instructions:
Your task is to convert a question into a SQL query, given a PostgreSQL database schema.
Adhere to these rules:
- **Only return the SQL query, nothing else**
- **Use proper PostgreSQL syntax**
- **Use table aliases to prevent ambiguity**
- **Always include LIMIT 5 to limit results**
- **Do not include explanations or comments**

### Database Schema:
{schema_sql}

### Question:
Generate a SQL query that answers: "{question}"

### Response:
"""

    try:
        logger.info(f"Generating SQL query for: {question}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:  # Reduced timeout
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
                    "temperature": 0.1,  # Low temperature for consistent output
                    "top_p": 0.9,
                    "num_predict": 200,  # Limit response length
                    "stop": ["###", "Explanation:", "Note:", "\n\n"]  # Stop tokens
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
                raise HTTPException(status_code=500, detail=f"AI service error: {response.status_code}")
            
            result = response.json()
            
            if "response" not in result:
                logger.error(f"Unexpected Ollama response format: {result}")
                raise HTTPException(status_code=500, detail="Invalid AI service response format")
            
            sql_query = result["response"].strip()
            
            if not sql_query:
                logger.error("Empty response from Ollama")
                # Fallback to simple query based on question
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
                if line.startswith(('--', '#', '/*')):  # Skip comments
                    continue
                if any(word in line.lower() for word in ['explanation:', 'note:', 'based on', 'this query']):
                    break  # Stop at explanations
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
                # Insert LIMIT before final semicolon
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating SQL query: {e}")
        return generate_fallback_query(question, schema)

def generate_fallback_query(question: str, schema: Dict[str, Any]) -> str:
    """Generate a simple fallback query when AI fails"""
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
    """Execute SQL query safely"""

    # Basic SQL injection protection
    sql_lower = sql_query.lower().strip()

    # Only allow SELECT statements
    if not sql_lower.startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous_keywords = [
        "insert", "update", "delete", "drop", "create", "alter",
        "truncate", "exec", "execute", "sp_", "xp_", "--", "/*", "*/"
    ]

    for keyword in dangerous_keywords:
        if keyword in sql_lower:
            raise HTTPException(status_code=400, detail=f"Query contains prohibited keyword: {keyword}")

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
                    if hasattr(value, 'isoformat'):  # datetime objects
                        value = value.isoformat()
                    elif hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool, type(None))):
                        value = str(value)
                    row_dict[column] = value
                rows.append(row_dict)

            return rows

    except SQLAlchemyError as e:
        logger.error(f"Database error executing query: {e}")
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

async def generate_explanation(question: str, sql_query: str, results: List[Dict[str, Any]]) -> str:
    """Generate explanation of the query and results with fallback"""
    
    # Simple fallback explanation to avoid another timeout
    results_summary = f"Query returned {len(results)} rows."
    if results and len(results) > 0:
        columns = list(results[0].keys())
        results_summary += f" Columns: {', '.join(columns[:3])}"  # Show first 3 columns
        if len(columns) > 3:
            results_summary += f" and {len(columns)-3} more."
    
    return f"The SQL query successfully retrieved data answering '{question}'. {results_summary}"

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check Ollama
        ollama_healthy = await check_ollama_health()
        
        return {
            "status": "healthy" if ollama_healthy else "partial",
            "database": "healthy",
            "ollama": "healthy" if ollama_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/schema", response_model=SchemaInfo)
async def get_schema():
    """Get database schema information"""
    schema = await get_database_schema()
    return SchemaInfo(**schema)

@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Process natural language query and return results"""
    try:
        logger.info(f"Processing query: {request.question}")

        # Get database schema
        schema = await get_database_schema()

        # Generate SQL query
        sql_query = await generate_sql_query(request.question, schema)
        logger.info(f"Generated SQL: {sql_query}")

        # Execute query
        results = await execute_sql_query(sql_query)
        logger.info(f"Query returned {len(results)} rows")

        # Generate explanation
        explanation = await generate_explanation(request.question, sql_query, results)

        response = QueryResponse(
            sql_query=sql_query,
            results=results,
            explanation=explanation,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "row_count": len(results),
                "question": request.question
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug/ollama")
async def debug_ollama():
    """Debug endpoint to check Ollama status"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if Ollama is running
            health_response = await client.get(f"{OLLAMA_URL}/api/tags")
            models = health_response.json()
            
            return {
                "ollama_url": OLLAMA_URL,
                "status": "healthy",
                "models": models.get("models", []),
                "response_status": health_response.status_code
            }
    except Exception as e:
        return {
            "ollama_url": OLLAMA_URL,
            "status": "unhealthy",
            "error": str(e),
            "suggestion": "Make sure Ollama is running with: docker-compose up ollama"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
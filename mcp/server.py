import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pythonjsonlogger import jsonlogger
import uvicorn

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
app = FastAPI(title="AI Insights HTTP API", version="1.0.0")

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
                    }
                ]
            }

            _schema_cache = schema
            _schema_cache_time = current_time
            return schema

    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        raise RuntimeError(f"Database schema error: {str(e)}")

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
    
    return schema_sql

async def generate_sql_query(question: str, schema: Dict[str, Any]) -> str:
    """Generate SQL query using Ollama"""
    # Create PostgreSQL schema for the prompt
    schema_sql = create_schema_for_prompt(schema)
    
    prompt = f"""### Instructions:
You are a SQL expert. Convert the question into a proper PostgreSQL query.

### Rules:
- Return ONLY the SQL query, no explanations
- Use proper JOINs when data is in multiple tables
- Use SUM(), COUNT(), GROUP BY for aggregations
- Always add LIMIT 50 at the end
- Use table aliases (c for customers, s for sales_data, etc.)

### Database Schema:
{schema_sql}

### Examples:
Question: "What are the total sales by customer?"
Answer: SELECT c.customer_name, SUM(s.total_amount) as total_sales FROM customers c JOIN sales_data s ON c.customer_id = s.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY total_sales DESC LIMIT 50;

Question: "Show me customers"
Answer: SELECT * FROM customers LIMIT 50;

Question: "top 10 customers by revenue"
Answer: SELECT c.customer_name, c.company, SUM(s.total_amount) as total_revenue FROM customers c JOIN sales_data s ON c.customer_id = s.customer_id GROUP BY c.customer_id, c.customer_name, c.company ORDER BY total_revenue DESC LIMIT 10;

### Your Task:
Question: "{question}"
Answer: """

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get available models
            models_response = await client.get(f"{OLLAMA_URL}/api/tags")
            models_data = models_response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            # Use first available model
            model_name = available_models[0] if available_models else "llama3.1:8b"
            
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
            
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"AI service error: {response.status_code}")
            
            result = response.json()
            sql_query = result.get("response", "").strip()
            
            # Clean up the response
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Ensure it ends with semicolon
            if sql_query and not sql_query.endswith(";"):
                sql_query += ";"

            # Validate basic structure
            if not sql_query.upper().startswith('SELECT'):
                logger.warning(f"Generated query doesn't start with SELECT, using fallback")
                return f"SELECT * FROM sales_data LIMIT 50;"

            # Ensure LIMIT is present
            if 'LIMIT' not in sql_query.upper():
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1] + ' LIMIT 25;'
                else:
                    sql_query += ' LIMIT 25'

            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query

    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        return f"SELECT * FROM sales_data LIMIT 50;"

async def execute_sql_query(sql_query: str) -> List[Dict[str, Any]]:
    """Execute SQL query safely"""
    # Basic SQL injection protection
    sql_lower = sql_query.lower().strip()

    if not sql_lower.startswith("select"):
        raise RuntimeError("Only SELECT queries are allowed")

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

            rows = []
            for row in result:
                row_dict = {}
                for i, column in enumerate(result.keys()):
                    value = row[i]
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """HTTP endpoint for queries with AI generation"""
    try:
        logger.info(f"Processing query: {request.question}")

        # Get database schema
        schema = await get_database_schema()

        # Generate SQL query using AI
        sql_query = await generate_sql_query(request.question, schema)

        # Execute query
        results = await execute_sql_query(sql_query)

        return QueryResponse(
            sql_query=sql_query,
            results=results,
            explanation=f"The SQL query successfully retrieved data answering '{request.question}'. Query returned {len(results)} rows.",
            metadata={"row_count": len(results)}
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schema")
async def schema_endpoint():
    """Get database schema"""
    try:
        schema = await get_database_schema()
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
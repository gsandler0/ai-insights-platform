import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
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
    title="AI Analytics Service",
    description="Intelligent analysis of query results",
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
engine = create_engine(DATABASE_URL)

class AnalysisType(str, Enum):
    TRENDS = "trends"
    ANOMALIES = "anomalies"  
    QUALITY = "quality"
    OPPORTUNITIES = "opportunities"
    AUTO = "auto"

class DataType(str, Enum):
    TIME_SERIES = "time_series"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    LIST = "list"
    MIXED = "mixed"

# Pydantic Models
class AnalysisRequest(BaseModel):
    query: str
    sql: str
    results: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    analysis_type: AnalysisType = AnalysisType.AUTO

class AnalysisInsight(BaseModel):
    type: str
    title: str
    description: str
    severity: Literal["info", "warning", "critical", "opportunity"]
    data: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    insights: List[AnalysisInsight]
    data_type: DataType
    summary: str
    metadata: Dict[str, Any]

class DataProfiler:
    """Analyze the structure and characteristics of result sets"""
    
    @staticmethod
    def profile_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"type": DataType.LIST, "row_count": 0}
        
        profile = {
            "row_count": len(results),
            "column_count": len(results[0].keys()),
            "columns": {},
            "data_types": {},
            "has_time_data": False,
            "has_numeric_data": False,
            "sample_row": results[0] if results else {},
            "column_names": list(results[0].keys()) if results else []
        }
        
        # Analyze each column
        for col_name in results[0].keys():
            values = [row.get(col_name) for row in results]
            non_null_values = [v for v in values if v is not None]
            
            profile["columns"][col_name] = {
                "null_count": len(values) - len(non_null_values),
                "unique_count": len(set(str(v) for v in non_null_values)),
                "sample_values": non_null_values[:3],
                "name": col_name  # Store the actual column name
            }
            
            # Detect data types
            if non_null_values:
                sample_value = non_null_values[0]
                
                # Check for dates/times
                if any(keyword in col_name.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                    profile["has_time_data"] = True
                    profile["data_types"][col_name] = "datetime"
                
                # Check for numeric data  
                elif isinstance(sample_value, (int, float)) or (isinstance(sample_value, str) and sample_value.replace('.', '').replace('-', '').isdigit()):
                    profile["has_numeric_data"] = True
                    profile["data_types"][col_name] = "numeric"
                    
                    # Store numeric statistics
                    numeric_values = []
                    for val in non_null_values:
                        try:
                            numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            continue
                    
                    if numeric_values:
                        profile["columns"][col_name].update({
                            "min_value": min(numeric_values),
                            "max_value": max(numeric_values),
                            "avg_value": sum(numeric_values) / len(numeric_values),
                            "is_numeric": True
                        })
                
                else:
                    profile["data_types"][col_name] = "categorical"
        
        return profile

    @staticmethod
    def determine_data_type(profile: Dict[str, Any]) -> DataType:
        """Determine the primary data type for analysis"""
        if profile["row_count"] <= 5:
            return DataType.LIST
        
        if profile["has_time_data"] and profile["has_numeric_data"]:
            return DataType.TIME_SERIES
            
        if profile["has_numeric_data"]:
            return DataType.NUMERICAL
            
        return DataType.CATEGORICAL

class QualityAnalyzer:
    """Analyze data quality issues"""
    
    @staticmethod
    async def analyze(results: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[AnalysisInsight]:
        insights = []
        
        # Check for missing data
        for col_name, col_info in profile["columns"].items():
            null_percentage = (col_info["null_count"] / profile["row_count"]) * 100
            
            if null_percentage > 50:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title=f"High Missing Data",
                    description=f"Column '{col_name}' has {null_percentage:.1f}% missing values ({col_info['null_count']} out of {profile['row_count']} records)",
                    severity="warning",
                    data={"column": col_name, "null_percentage": null_percentage, "affected_records": col_info["null_count"]}
                ))
            elif null_percentage > 20:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title=f"Missing Data Detected",
                    description=f"Column '{col_name}' has {null_percentage:.1f}% missing values ({col_info['null_count']} records)",
                    severity="info",
                    data={"column": col_name, "null_percentage": null_percentage, "affected_records": col_info["null_count"]}
                ))
        
        # Check for duplicates
        if profile["row_count"] > 1:
            unique_rows = len(set(json.dumps(row, sort_keys=True) for row in results))
            duplicate_count = profile["row_count"] - unique_rows
            duplicate_percentage = (duplicate_count / profile["row_count"]) * 100
            
            if duplicate_percentage > 0:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title="Duplicate Records Found",
                    description=f"Found {duplicate_count} duplicate records ({duplicate_percentage:.1f}% of total data)",
                    severity="warning" if duplicate_percentage > 10 else "info",
                    data={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_percentage, "unique_rows": unique_rows}
                ))
        
        # Check for low cardinality in potential key columns
        for col_name, col_info in profile["columns"].items():
            uniqueness_ratio = col_info["unique_count"] / (profile["row_count"] - col_info["null_count"]) if (profile["row_count"] - col_info["null_count"]) > 0 else 0
            
            if uniqueness_ratio < 0.1 and col_info["unique_count"] < 5:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title="Low Data Diversity",
                    description=f"Column '{col_name}' has only {col_info['unique_count']} unique values across {profile['row_count']} records",
                    severity="info",
                    data={"column": col_name, "unique_count": col_info["unique_count"], "uniqueness_ratio": uniqueness_ratio}
                ))
        
        return insights

class TrendAnalyzer:
    """Analyze trends in time series and numerical data"""
    
    @staticmethod  
    async def analyze(results: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[AnalysisInsight]:
        insights = []
        
        if profile["row_count"] < 3:
            return insights
            
        # Find numeric columns for trend analysis
        numeric_columns = [col for col, dtype in profile["data_types"].items() if dtype == "numeric"]
        
        for col_name in numeric_columns:
            values = []
            for row in results:
                val = row.get(col_name)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 3:
                col_info = profile["columns"][col_name]
                
                # Calculate trend direction
                increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
                decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
                total_transitions = len(values) - 1
                
                if increasing > total_transitions * 0.7:
                    trend_strength = (increasing / total_transitions) * 100
                    value_range = f"{col_info.get('min_value', 'N/A'):.2f} to {col_info.get('max_value', 'N/A'):.2f}"
                    insights.append(AnalysisInsight(
                        type="trend",
                        title=f"Upward Trend in {col_name}",
                        description=f"Column '{col_name}' shows a strong upward trend ({trend_strength:.0f}% of transitions increasing). Values range from {value_range}.",
                        severity="opportunity",
                        data={
                            "column": col_name, 
                            "trend": "increasing", 
                            "strength": trend_strength,
                            "min_value": col_info.get("min_value"),
                            "max_value": col_info.get("max_value"),
                            "avg_value": col_info.get("avg_value")
                        }
                    ))
                
                elif decreasing > total_transitions * 0.7:
                    trend_strength = (decreasing / total_transitions) * 100
                    value_range = f"{col_info.get('max_value', 'N/A'):.2f} to {col_info.get('min_value', 'N/A'):.2f}"
                    insights.append(AnalysisInsight(
                        type="trend", 
                        title=f"Downward Trend in {col_name}",
                        description=f"Column '{col_name}' shows a consistent downward trend ({trend_strength:.0f}% of transitions decreasing). Values declining from {value_range}.",
                        severity="warning",
                        data={
                            "column": col_name, 
                            "trend": "decreasing", 
                            "strength": trend_strength,
                            "min_value": col_info.get("min_value"),
                            "max_value": col_info.get("max_value"),
                            "avg_value": col_info.get("avg_value")
                        }
                    ))
                
                # Check for high variability
                if len(values) > 2 and col_info.get("max_value") and col_info.get("min_value"):
                    value_range = col_info["max_value"] - col_info["min_value"]
                    avg_value = col_info.get("avg_value", 0)
                    if avg_value > 0:
                        coefficient_of_variation = (value_range / avg_value) * 100
                        if coefficient_of_variation > 200:  # High variability
                            insights.append(AnalysisInsight(
                                type="trend",
                                title=f"High Variability in {col_name}",
                                description=f"Column '{col_name}' shows high variability (range: {value_range:.2f}, average: {avg_value:.2f}). Consider investigating outliers or data collection issues.",
                                severity="info",
                                data={
                                    "column": col_name,
                                    "variability": "high",
                                    "coefficient_of_variation": coefficient_of_variation,
                                    "range": value_range,
                                    "average": avg_value
                                }
                            ))
        
        return insights

class AnomalyAnalyzer:
    """Detect anomalies and outliers"""
    
    @staticmethod
    async def analyze(results: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[AnalysisInsight]:
        insights = []
        
        if profile["row_count"] < 5:
            return insights
            
        # Find numeric columns for anomaly detection
        numeric_columns = [col for col, dtype in profile["data_types"].items() if dtype == "numeric"]
        
        for col_name in numeric_columns:
            values = []
            outlier_rows = []
            
            for i, row in enumerate(results):
                val = row.get(col_name)
                if val is not None:
                    try:
                        float_val = float(val)
                        values.append(float_val)
                        outlier_rows.append((i, float_val, row))
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 5:
                col_info = profile["columns"][col_name]
                
                # Simple outlier detection using IQR
                sorted_values = sorted(values)
                q1_idx = len(sorted_values) // 4
                q3_idx = 3 * len(sorted_values) // 4
                q1 = sorted_values[q1_idx]
                q3 = sorted_values[q3_idx] 
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [(val, row) for _, val, row in outlier_rows if val < lower_bound or val > upper_bound]
                    
                    if outliers:
                        outlier_percentage = (len(outliers) / len(values)) * 100
                        outlier_values = [val for val, _ in outliers[:3]]  # Show first 3
                        
                        # Create more descriptive message
                        outlier_description = f"Column '{col_name}' contains {len(outliers)} outlier values ({outlier_percentage:.1f}% of data). "
                        outlier_description += f"Normal range: {lower_bound:.2f} to {upper_bound:.2f}. "
                        outlier_description += f"Outlier examples: {', '.join([f'{v:.2f}' for v in outlier_values])}"
                        
                        insights.append(AnalysisInsight(
                            type="anomaly",
                            title=f"Outliers Detected in {col_name}",
                            description=outlier_description,
                            severity="warning" if outlier_percentage > 10 else "info",
                            data={
                                "column": col_name,
                                "outlier_count": len(outliers),
                                "outlier_percentage": outlier_percentage,
                                "outlier_examples": outlier_values,
                                "normal_range": {"lower": lower_bound, "upper": upper_bound},
                                "q1": q1,
                                "q3": q3
                            }
                        ))
        
        # Check for categorical anomalies
        categorical_columns = [col for col, dtype in profile["data_types"].items() if dtype == "categorical"]
        
        for col_name in categorical_columns:
            values = [str(row.get(col_name, '')) for row in results if row.get(col_name) is not None]
            
            if len(values) > 0:
                from collections import Counter
                value_counts = Counter(values)
                total_count = len(values)
                
                # Find very rare values (less than 2% of data)
                rare_values = [(val, count) for val, count in value_counts.items() if (count / total_count) < 0.02 and count == 1]
                
                if len(rare_values) > 0 and len(rare_values) < total_count * 0.1:  # Don't flag if most values are unique
                    rare_examples = [val for val, _ in rare_values[:3]]
                    insights.append(AnalysisInsight(
                        type="anomaly",
                        title=f"Rare Values in {col_name}",
                        description=f"Column '{col_name}' has {len(rare_values)} rare values that appear only once. Examples: {', '.join(rare_examples)}",
                        severity="info",
                        data={
                            "column": col_name,
                            "rare_value_count": len(rare_values),
                            "rare_examples": rare_examples,
                            "total_unique_values": len(value_counts)
                        }
                    ))
        
        return insights

class AnalyticsEngine:
    """Main analytics orchestrator"""
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.quality_analyzer = QualityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_analyzer = AnomalyAnalyzer()
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform comprehensive analysis of query results"""
        
        if not request.results:
            return AnalysisResponse(
                insights=[],
                data_type=DataType.LIST,
                summary="No data to analyze",
                metadata={"row_count": 0}
            )
        
        # Profile the data
        profile = self.profiler.profile_results(request.results)
        data_type = self.profiler.determine_data_type(profile)
        
        logger.info(f"Analyzing {profile['row_count']} rows of {data_type} data with columns: {profile['column_names']}")
        
        # Collect insights based on data type and analysis type
        all_insights = []
        
        if request.analysis_type in [AnalysisType.AUTO, AnalysisType.QUALITY]:
            quality_insights = await self.quality_analyzer.analyze(request.results, profile)
            all_insights.extend(quality_insights)
        
        if request.analysis_type in [AnalysisType.AUTO, AnalysisType.TRENDS] and data_type in [DataType.TIME_SERIES, DataType.NUMERICAL]:
            trend_insights = await self.trend_analyzer.analyze(request.results, profile) 
            all_insights.extend(trend_insights)
        
        if request.analysis_type in [AnalysisType.AUTO, AnalysisType.ANOMALIES] and data_type in [DataType.TIME_SERIES, DataType.NUMERICAL]:
            anomaly_insights = await self.anomaly_analyzer.analyze(request.results, profile)
            all_insights.extend(anomaly_insights)
        
        
        structured_insights = []
        for i, insight in enumerate(all_insights):
            if isinstance(insight, str):
                structured_insights.append({
                    "title": f"Insight {i+1}",
                    "description": insight,
                    "severity": "info",
                    "type": "general"
                })
            elif isinstance(insight, dict) or hasattr(insight, "__dict__"):
                # Convert dataclass/object to dict if needed
                data = insight if isinstance(insight, dict) else insight.__dict__
                structured_insights.append({
                    "title": data.get("title") or f"Insight {i+1}",
                    "description": data.get("description") or data.get("message", "No description"),
                    "severity": data.get("severity", "info"),
                    "type": data.get("type", "general")
                })


        # Generate summary
        summary = self._generate_summary(all_insights, profile, data_type)
        
        return AnalysisResponse(
            insights=structured_insights,
            data_type=getattr(data_type, 'value', data_type),
            summary=summary or "No summary generated",
            metadata={
                "row_count": profile.get("row_count", 0),
                "column_count": profile.get("column_count", 0),
                "analysis_time": datetime.now().isoformat(),
                "data_profile": profile,
                "columns_analyzed": profile.get("column_names", [])
            }
)   
    
    def _generate_summary(self, insights: List[AnalysisInsight], profile: Dict[str, Any], data_type: DataType) -> str:
        """Generate a human-readable summary"""
        
        columns_text = f"({', '.join(profile['column_names'])})" if profile['column_names'] else ""
        
        if not insights:
            return f"Analyzed {profile['row_count']} rows of {data_type.value} data {columns_text}. No significant patterns or issues detected."
        
        summary_parts = [f"Analysis of {profile['row_count']} rows {columns_text} revealed:"]
        
        # Group insights by type
        by_type = {}
        for insight in insights:
            if insight.type not in by_type:
                by_type[insight.type] = []
            by_type[insight.type].append(insight)
        
        for insight_type, type_insights in by_type.items():
            if insight_type == "data_quality":
                affected_columns = list(set([insight.data.get("column", "unknown") for insight in type_insights if insight.data]))
                summary_parts.append(f"• {len(type_insights)} data quality issues affecting columns: {', '.join(affected_columns)}")
            elif insight_type == "trend":
                trend_columns = list(set([insight.data.get("column", "unknown") for insight in type_insights if insight.data]))
                summary_parts.append(f"• {len(type_insights)} trend patterns in columns: {', '.join(trend_columns)}")  
            elif insight_type == "anomaly":
                anomaly_columns = list(set([insight.data.get("column", "unknown") for insight in type_insights if insight.data]))
                summary_parts.append(f"• {len(type_insights)} anomalies detected in columns: {', '.join(anomaly_columns)}")
        
        return " ".join(summary_parts)

# Initialize the analytics engine
analytics_engine = AnalyticsEngine()

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_results(request: AnalysisRequest):
    """Analyze query results for insights"""
    try:
        logger.info(f"Analyzing query: {request.query[:100]}...")
        
        response = await analytics_engine.analyze(request)
        
        logger.info(f"Generated {len(response.insights)} insights for columns: {response.metadata.get('columns_analyzed', [])}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
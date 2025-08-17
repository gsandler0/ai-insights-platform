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
            "sample_row": results[0] if results else {}
        }
        
        # Analyze each column
        for col_name in results[0].keys():
            values = [row.get(col_name) for row in results]
            non_null_values = [v for v in values if v is not None]
            
            profile["columns"][col_name] = {
                "null_count": len(values) - len(non_null_values),
                "unique_count": len(set(str(v) for v in non_null_values)),
                "sample_values": non_null_values[:3]
            }
            
            # Detect data types
            if non_null_values:
                sample_value = non_null_values[0]
                
                # Check for dates/times
                if any(keyword in col_name.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    profile["has_time_data"] = True
                    profile["data_types"][col_name] = "datetime"
                
                # Check for numeric data  
                elif isinstance(sample_value, (int, float)) or (isinstance(sample_value, str) and sample_value.replace('.', '').replace('-', '').isdigit()):
                    profile["has_numeric_data"] = True
                    profile["data_types"][col_name] = "numeric"
                
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
                    title=f"High Missing Data in {col_name}",
                    description=f"{null_percentage:.1f}% of values are missing in column '{col_name}'",
                    severity="warning",
                    data={"column": col_name, "null_percentage": null_percentage}
                ))
            elif null_percentage > 20:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title=f"Missing Data in {col_name}",
                    description=f"{null_percentage:.1f}% of values are missing in column '{col_name}'",
                    severity="info",
                    data={"column": col_name, "null_percentage": null_percentage}
                ))
        
        # Check for duplicates
        if profile["row_count"] > 1:
            unique_rows = len(set(json.dumps(row, sort_keys=True) for row in results))
            duplicate_percentage = ((profile["row_count"] - unique_rows) / profile["row_count"]) * 100
            
            if duplicate_percentage > 0:
                insights.append(AnalysisInsight(
                    type="data_quality",
                    title="Duplicate Records Found",
                    description=f"{duplicate_percentage:.1f}% of records appear to be duplicates",
                    severity="warning" if duplicate_percentage > 10 else "info",
                    data={"duplicate_percentage": duplicate_percentage, "unique_rows": unique_rows}
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
            values = [float(row[col_name]) for row in results if row.get(col_name) is not None and str(row[col_name]).replace('.', '').replace('-', '').isdigit()]
            
            if len(values) >= 3:
                # Simple trend detection
                if len(values) > 1:
                    increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
                    decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
                    
                    if increasing > len(values) * 0.7:
                        trend_strength = (increasing / (len(values) - 1)) * 100
                        insights.append(AnalysisInsight(
                            type="trend",
                            title=f"Upward Trend in {col_name}",
                            description=f"Strong upward trend detected ({trend_strength:.0f}% of periods increasing)",
                            severity="opportunity",
                            data={"column": col_name, "trend": "increasing", "strength": trend_strength}
                        ))
                    
                    elif decreasing > len(values) * 0.7:
                        trend_strength = (decreasing / (len(values) - 1)) * 100
                        insights.append(AnalysisInsight(
                            type="trend", 
                            title=f"Downward Trend in {col_name}",
                            description=f"Consistent downward trend detected ({trend_strength:.0f}% of periods decreasing)",
                            severity="warning",
                            data={"column": col_name, "trend": "decreasing", "strength": trend_strength}
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
            for row in results:
                val = row.get(col_name)
                if val is not None and str(val).replace('.', '').replace('-', '').isdigit():
                    values.append(float(val))
            
            if len(values) >= 5:
                # Simple outlier detection using IQR
                sorted_values = sorted(values)
                q1 = sorted_values[len(sorted_values)//4]
                q3 = sorted_values[3*len(sorted_values)//4] 
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    
                    if outliers:
                        outlier_percentage = (len(outliers) / len(values)) * 100
                        insights.append(AnalysisInsight(
                            type="anomaly",
                            title=f"Outliers Detected in {col_name}",
                            description=f"{len(outliers)} outlier values found ({outlier_percentage:.1f}% of data)",
                            severity="warning" if outlier_percentage > 10 else "info",
                            data={
                                "column": col_name,
                                "outlier_count": len(outliers),
                                "outlier_percentage": outlier_percentage,
                                "outliers": outliers[:3]  # Show first 3
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
        
        logger.info(f"Analyzing {profile['row_count']} rows of {data_type} data")
        
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
        
        # Generate summary
        summary = self._generate_summary(all_insights, profile, data_type)
        
        return AnalysisResponse(
            insights=all_insights,
            data_type=data_type,
            summary=summary,
            metadata={
                "row_count": profile["row_count"],
                "column_count": profile["column_count"],
                "analysis_time": datetime.now().isoformat(),
                "data_profile": profile
            }
        )
    
    def _generate_summary(self, insights: List[AnalysisInsight], profile: Dict[str, Any], data_type: DataType) -> str:
        """Generate a human-readable summary"""
        
        if not insights:
            return f"Analyzed {profile['row_count']} rows of {data_type.value} data. No significant patterns or issues detected."
        
        summary_parts = [f"Analysis of {profile['row_count']} rows revealed:"]
        
        # Group insights by type
        by_type = {}
        for insight in insights:
            if insight.type not in by_type:
                by_type[insight.type] = []
            by_type[insight.type].append(insight)
        
        for insight_type, type_insights in by_type.items():
            if insight_type == "data_quality":
                summary_parts.append(f"• {len(type_insights)} data quality considerations")
            elif insight_type == "trend":
                summary_parts.append(f"• {len(type_insights)} trend patterns identified")  
            elif insight_type == "anomaly":
                summary_parts.append(f"• {len(type_insights)} anomalies detected")
        
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
        
        logger.info(f"Generated {len(response.insights)} insights")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

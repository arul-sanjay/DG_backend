
import os
import json
import logging
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from pydantic import BaseModel
from pipeline import DataGovernancePipeline
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")

app = FastAPI(title="Data Governance API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineParams(BaseModel):
    sample_size: int = 1000

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), sample_size: int = Form(1000)):
    """
    Upload a file and run the data governance pipeline.
    """
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    try:
        # Initialize pipeline
        pipeline = DataGovernancePipeline(GROQ_API_KEY, sample_size=sample_size)
        
        # Process the file
        contents = await file.read()
        file_object = io.BytesIO(contents)
        file_object.name = file.filename
        
        # Run pipeline steps
        df = pipeline.step1_dataset_ingestion(file_object)
        initial_analysis = pipeline.step2_initial_analysis(df)
        quality_report = pipeline.step3_data_quality_assessment()
        bias_results = pipeline.step4_bias_detection()
        privacy_results = pipeline.step5_privacy_assessment()
        lineage_info = pipeline.step6_lineage_documentation()
        compliance_report = pipeline.step7_governance_compliance()
        recommendations = pipeline.step8_recommendations()
        
        # Generate full report
        full_report = json.loads(pipeline.generate_full_report())
        
        # Include dataset preview in response
        dataset_preview = df.head(10).to_dict(orient="records")
        dataset_stats = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist()
        }
        
        return {
            "message": "Pipeline completed successfully",
            "dataset_preview": dataset_preview,
            "dataset_stats": dataset_stats,
            "initial_analysis": initial_analysis,
            "quality_report": quality_report,
            "bias_results": bias_results,
            "privacy_results": privacy_results,
            "lineage_info": lineage_info,
            "compliance_report": compliance_report,
            "recommendations": recommendations,
            "full_report": full_report
        }
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

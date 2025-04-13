
import pandas as pd
from langchain_groq import ChatGroq
from utils import load_dataset, sample_dataframe, check_unique_columns, convert_to_json_serializable
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional, Any, Union
import json
import logging
import io

logger = logging.getLogger(__name__)

# Pydantic Models for Structured Outputs
class InitialAnalysis(BaseModel):
    """Model for initial dataset analysis."""
    domain: str
    column_meanings: Dict[str, str]

class Recommendation(BaseModel):
    """Model for a single recommendation."""
    category: str
    description: str

class Recommendations(BaseModel):
    """Model for a list of recommendations."""
    recommendations: List[Recommendation]

class DataGovernancePipeline:
    """Automated data governance pipeline using Groq's Llama 3.1 70B model."""

    def __init__(self, api_key: str, sample_size: int = 1000):
        """
        Initialize the pipeline with Groq's LLM and configuration.

        Args:
            api_key (str): Groq API key for LLM access.
            sample_size (int): Number of rows to sample from large datasets.
        """
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
        self.sample_size: int = sample_size
        self.df: Optional[pd.DataFrame] = None
        self.results: Dict = {}
        self.file_name: str = ""
        self.upload_time: str = ""

    def step1_dataset_ingestion(self, file: Union[io.BytesIO, Any]) -> pd.DataFrame:
        """Load and sample the dataset."""
        self.file_name = file.name
        self.upload_time = pd.Timestamp.now().isoformat()
        self.df = load_dataset(file)
        self.df = sample_dataframe(self.df, self.sample_size)
        logger.info(f"Dataset ingested: {self.df.shape}")
        return self.df

    def step2_initial_analysis(self, df: pd.DataFrame) -> Dict:
        """Infer domain and column meanings using Groq's Llama 3.1 70B."""
        columns = df.columns.tolist()
        sample_data = df.head(5).to_dict(orient="records")
        prompt = (
            f"""You are an AI assistant tasked with analyzing a dataset named '{self.file_name}', which contains employee-related data, likely for HR purposes like managing personnel records. Below are the column names and a sample of the data. Your job is to infer the domain of the dataset (e.g., HR, finance, sales) and the meaning of each column based on the column names and sample data provided.

Columns: {columns}

Sample Data: {sample_data}

Based on the column names and sample data, provide a JSON object with two keys: 'domain' (a string indicating the inferred domain) and 'column_meanings' (a dictionary where each key is a column name and each value is a string describing what that column likely represents). Do not return 'Unknown'. Make your best guess based on common patterns. For example, for columns like ['CustomerID', 'PurchaseAmount', 'PurchaseDate'], you might return:
{{
  "domain": "Sales",
  "column_meanings": {{
    "CustomerID": "Unique identifier for each customer",
    "PurchaseAmount": "The amount spent by the customer in a transaction",
    "PurchaseDate": "The date when the purchase was made"
  }}
}}
For this dataset, infer similarly (e.g., 'Education' might be the employee's education level). Ensure your response is a valid JSON object with no additional text outside it."""
        )
        response = self.llm.invoke(prompt).content
        logger.info(f"LLM response for initial analysis: {response}")
        try:
            analysis_dict = json.loads(response)
            analysis = InitialAnalysis(**analysis_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse initial analysis: {str(e)}")
            analysis = InitialAnalysis(domain="Unknown", column_meanings={col: "Unknown" for col in columns})
        self.results["initial_analysis"] = analysis.dict()
        return analysis.dict()

    def step3_data_quality_assessment(self) -> Dict:
        """Assess data quality using Groq's Llama 3.1 70B."""
        missing_values = {col: int(val) for col, val in self.df.isnull().sum().items()}
        duplicates = int(self.df.duplicated().sum())
        prompt = (
            "The dataset has the following missing values per column: "
            f"{json.dumps(missing_values)}, and {duplicates} duplicate rows. "
            "Provide detailed suggestions for improving data quality in a concise manner."
            "it is important that you should give the content in proper markdown format"
        )
        suggestions = self.llm.invoke(prompt).content
        logger.info(f"LLM suggestions for data quality: {suggestions}")
        report = {
            "missing_values": missing_values,
            "duplicates": duplicates,
            "suggestions": suggestions
        }
        self.results["quality_report"] = report
        logger.info("Data quality assessment completed")
        return report

    def step4_bias_detection(self) -> Dict:
        """Detect potential biases using Groq's Llama 3.1 70B."""
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        distributions = {col: self.df[col].value_counts(normalize=True).to_dict() for col in categorical_cols}
        prompt = (
            "The dataset has the following distributions for categorical columns: "
            f"{json.dumps(distributions, indent=2)}. Analyze these distributions for potential biases "
            "(e.g., skewed distributions, underrepresented groups) and provide detailed insights."
            "your response should be in markdown format"
        )
        analysis = self.llm.invoke(prompt).content
        report = {
            "distributions": distributions,
            "analysis": analysis
        }
        self.results["bias_results"] = report
        logger.info("Bias detection completed")
        return report

    def step5_privacy_assessment(self) -> Dict:
        """Assess privacy risks using Groq's Llama 3.1 70B."""
        pii_keywords = [
            "id", "name", "email", "phone", "address", "ssn", "credit", "account", "age",
            "gender", "city", "birth", "date", "dob", "social", "security"
        ]
        keyword_pii = [col for col in self.df.columns if any(kw in col.lower() for kw in pii_keywords)]
        unique_pii = check_unique_columns(self.df)
        pii_columns = list(set(keyword_pii + unique_pii))
        prompt = (
            "The dataset has the following potential PII columns based on names and uniqueness: "
            f"{pii_columns}. Suggest specific handling mechanisms (e.g., masking, encryption, removal) "
            "to ensure privacy compliance with regulations like GDPR or CCPA."
            "your response should be in markdown format"
        )
        suggestions = self.llm.invoke(prompt).content
        report = {
            "pii_columns": pii_columns,
            "suggestions": suggestions
        }
        self.results["privacy_results"] = report
        logger.info("Privacy assessment completed")
        return report

    def step6_lineage_documentation(self) -> Dict:
        """Document data lineage."""
        lineage_info = {
            "source": self.file_name,
            "upload_time": self.upload_time,
            "sample_size": self.sample_size,
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "timestamp": pd.Timestamp.now().isoformat(),
            "transformations": ["Sampled"]
        }
        self.results["lineage_info"] = lineage_info
        logger.info("Lineage documentation completed")
        return lineage_info

    def step7_governance_compliance(self) -> Dict:
        """Check compliance with predefined policies."""
        policies = {
            "max_missing_values": 0.1,  # Max 10% missing values per column
            "no_pii_in_open_access": True
        }
        missing_ratio = self.df.isnull().mean()
        compliance = {
            "missing_values_check": all(ratio <= policies["max_missing_values"] for ratio in missing_ratio.values),
            "pii_check": not any(self.results.get("privacy_results", {}).get("pii_columns", [])) if policies["no_pii_in_open_access"] else True
        }
        self.results["compliance_report"] = compliance
        logger.info("Governance compliance completed")
        return compliance

    def step8_recommendations(self) -> Dict:
        """Generate recommendations using Grok's Llama model."""
        # Convert results to a JSON-serializable format
        serializable_results = convert_to_json_serializable(self.results)
        
        # Stricter prompt to ensure only a valid JSON object is returned
        prompt = (
            "Based on the following analysis results, provide actionable recommendations "
            "for improving data quality and governance. **Your response must consist solely of a valid JSON object** "
            "with the following structure: {\"recommendations\": [{\"category\": \"string\", \"description\": \"string\"}, ...]}. "
            "Do not include any additional text, comments, or explanations outside the JSON object. "
            "Ensure the JSON is properly formatted and can be directly parsed. "
            "For example, if PII columns are detected, your response should be: "
            "{\"recommendations\": [{\"category\": \"Privacy\", \"description\": \"Anonymize or remove PII columns like Age\"}]}. "
            "If no specific issues are found, provide a general recommendation.\n"
            f"{json.dumps(serializable_results, indent=2)}"
        )
        
        response = self.llm.invoke(prompt).content
        logger.info(f"Raw LLM response for recommendations: {response}")
        
        # Check if the response is empty
        if not response.strip():
            logger.error("LLM returned an empty response")
            self.results["recommendations"] = {
                "recommendations": [{"category": "General", "description": "No specific recommendations available. Review the dataset for potential issues."}]
            }
        else:
            try:
                # Attempt to parse the LLM's response into JSON
                recommendations_dict = json.loads(response)
                # Validate the structure
                recommendations = Recommendations(**recommendations_dict)
                self.results["recommendations"] = recommendations.dict()
            except (json.JSONDecodeError, ValidationError) as e:
                # If parsing fails, log the error and provide a default recommendation
                logger.error(f"Failed to parse recommendations: {str(e)}")
                self.results["recommendations"] = {
                    "recommendations": [{"category": "General", "description": "Review the dataset for potential issues like PII or missing data."}]
                }
        
        logger.info("Recommendations generated")
        return self.results["recommendations"]

    def generate_full_report(self) -> str:
        """Generate a JSON report of all results."""
        serializable_results = convert_to_json_serializable(self.results)
        report = json.dumps(serializable_results, indent=2)
        logger.debug(f"Full report generated: {report[:200]}...")
        return report

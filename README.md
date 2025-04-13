
# Data Governance Pipeline API

This is a FastAPI backend for the Data Governance Pipeline application. It provides an API endpoint for uploading and analyzing datasets with AI-powered governance assessment.

## Requirements

- Python 3.8+
- FastAPI
- pandas
- langchain_groq
- pydantic
- other dependencies listed in requirements.txt

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set the GROQ API key as an environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Or use the default key provided in the code (not recommended for production).

## Running the Server

Run the development server:

```bash
python run.py
```

The API will be available at http://localhost:8000.

## API Endpoints

### POST /api/upload

Upload a dataset file (CSV or Excel) for analysis.

**Parameters:**
- `file`: The dataset file to upload (required)
- `sample_size`: Number of rows to sample from large datasets (default: 1000)

**Response:**
JSON object containing:
- Dataset preview and statistics
- Initial analysis of domain and column meanings
- Data quality assessment
- Bias detection results
- Privacy assessment
- Lineage documentation
- Governance compliance report
- Recommendations for improvement
- Full analysis report

## Testing the API

You can test the API using curl:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/your/dataset.csv" \
  -F "sample_size=1000"
```

Or use the React frontend application that connects to this API.

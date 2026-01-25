from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Union, List
import subprocess
import json
from pathlib import Path

# Import your prediction logic
# Ensure src/main.py exists and has predict_main
from src.main import predict_main 

app = FastAPI(title="Loan-model API", version="1.0.0")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class LoanRequest(BaseModel):
    person_income: float
    person_home_ownership: str
    loan_int_rate: float
    loan_percent_income: float
    previous_loan_defaults_on_file: str
    loan_amnt: float 

class TrainingParams(BaseModel):
    # Using specific defaults ensures the model is trained consistently
    n_estimators: Optional[int] = Field(100, description="Number of trees")
    max_depth: Optional[int] = Field(15, description="Max depth to prevent overfitting")
    min_samples_split: Optional[int] = 20
    max_features: Optional[Union[str, int, float]] = "sqrt"
    
    # Allow Dict (custom weights) or str ("balanced")
    class_weight: Optional[Union[Dict[int, float], str]] = None

# --- Helper Functions ---

def run_make_command(args: List[str]) -> dict:
    """
    Execute a Makefile target safely.
    Args:
        args: List of command arguments, e.g., ["train", "PARAMS={...}"]
    """
    try:
        # Prepend 'make' to the arguments
        command = ["make"] + args
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        # This catches errors if the make command returns a non-zero exit code
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Make command failed",
                "command": " ".join(command),
                "error_output": e.stderr,
                "standard_output": e.stdout
            }
        )
    except Exception as e:
        # This catches general system errors (e.g., make not found)
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoints ---
@app.get("/")
def read_root():
    # Ensure this file exists, otherwise return a simple message
    index_path = Path("FrontEnd/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Loan Model API is running"}

@app.get("/health")
def health_check():
    """Check if model file exists"""
    model_path = Path("models/trained_model.pkl")
    return {
        "status": "healthy",
        "model_ready": model_path.exists(),
        "model_path": str(model_path)
    }

@app.post("/predict") 
def predict_loan_default(input_data: LoanRequest):
    try:
        # Ensure the model exists before trying to predict
        if not Path("models/trained_model.pkl").exists():
            raise HTTPException(status_code=503, detail="Model not loaded or trained yet.")

        prediction = predict_main(input_data, model_path="models/trained_model.pkl")
        return JSONResponse(content={"prediction": int(prediction[0])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrain")
def train_model_endpoint(params: TrainingParams):
    """
    Trigger model training via Makefile with custom hyperparameters.
    """
    try:
        # 1. Convert params to dict
        # We DO NOT use exclude_unset=True. 
        # We want to send the full default configuration to ensure the 
        # Python script receives all optimized parameters (e.g. max_depth=15).
        params_dict = params.dict()
        
        # 2. Dump to Compact JSON (no spaces)
        # This prevents issues with Make/Shell splitting arguments on spaces.
        params_json = json.dumps(params_dict, separators=(',', ':'))

        # 3. Construct arguments list
        # We pass the target and the variable as separate arguments.
        # Format: make train PARAMS='{"key":"value"}'
        # Note: We don't need extra quotes around params_json here because
        # subprocess.run handles the argument boundaries safely.
        make_args = ["train", f"PARAMS='{params_json}'"]
        
        print(f"Triggering training with: make {' '.join(make_args)}")
        
        # 4. Run Command
        result = run_make_command(make_args)

        return JSONResponse(content={
            "message": "Model training completed successfully",
            "model_file": "models/trained_model.pkl",
            "used_params": params_dict,
            "details": result.get("stdout")
        })

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            content={"message": "Unexpected error triggering training", "error": str(e)}
        )
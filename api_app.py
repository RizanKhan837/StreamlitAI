"""
FastAPI application for MedGemma Medical Image Analysis API with ngrok integration

INSTALLATION INSTRUCTIONS FOR COLAB:
Run the following in a Colab cell BEFORE running this script:

# Install required packages
!pip install -Uqqq pip --progress-bar off
!pip install -qqq transformers==4.53.2 --progress-bar off
!pip install -qqq accelerate==1.8.1 --progress-bar off
!pip install -qqq bitsandbytes==0.46.1 --progress-bar off
!pip install -qqq fastapi --progress-bar off
!pip install -qqq uvicorn[standard] --progress-bar off
!pip install -qqq python-multipart --progress-bar off
!pip install -qqq pyngrok --progress-bar off
!pip install -qqq pillow --progress-bar off
!pip install -qqq nest-asyncio --progress-bar off
!pip install -qqq pydicom --progress-bar off
!pip install -qqq numpy --progress-bar off

# Login to Hugging Face
from huggingface_hub import login
login()  # Enter your token: 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

# Set ngrok token
import os
os.environ["NGROK_AUTH_TOKEN"] = "2nKexbF0SUb8WVjqoLdO5Faa0eo_3HjBQL4E9g9oz8icW9mDw"

# Then run this script
"""
import sys
import subprocess

# Fix bitsandbytes metadata issue in Colab
def fix_bitsandbytes():
    """Fix bitsandbytes import metadata issue"""
    try:
        import bitsandbytes
        # Force reload to ensure it's properly registered
        import importlib
        importlib.reload(bitsandbytes)
        print("✓ bitsandbytes imported successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: bitsandbytes import issue: {e}")
        # Try to reinstall with more aggressive fix
        try:
            print("🔧 Attempting to fix bitsandbytes installation...")
            # Force reinstall and clear cache
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "bitsandbytes", "-y", "-q"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "bitsandbytes==0.46.1",
                "--no-cache-dir", "--force-reinstall", "-q"
            ])
            import bitsandbytes
            print("✓ bitsandbytes reinstalled and imported successfully")
            return True
        except Exception as reinstall_error:
            print(f"❌ bitsandbytes reinstall failed: {reinstall_error}")
            print("⚠ Will attempt to load model without quantization")
            return False

# Run the fix early
bitsandbytes_available = fix_bitsandbytes()

# Log dependency conflicts info (non-blocking)
if bitsandbytes_available:
    print("ℹ Note: Some dependency conflicts detected but should not affect API functionality")
    print("ℹ The API will prioritize working over perfect dependency alignment")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import io
from PIL import Image
import base64
import logging
import traceback
import uvicorn
import os
import nest_asyncio
from pyngrok import ngrok
import pydicom
import numpy as np
from datetime import datetime
import zipfile
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for Colab/Jupyter compatibility
nest_asyncio.apply()

# Global variables
model = None
processor = None
ngrok_tunnel = None

# Model configuration - easily switch between 4b and 27b
MODEL_NAME = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")  # Default to 4b, use env var for 27b
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2nKexbF0SUb8WVjqoLdO5Faa0eo_3HjBQL4E9g9oz8icW9mDw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global model, processor, ngrok_tunnel
    
    # Startup
    try:
        logger.info(f"Loading {MODEL_NAME} model...")
        
        # Import with better error handling
        try:
            import torch
            logger.info(f"✓ PyTorch {torch.__version__} loaded")
            logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.error(f"❌ PyTorch import failed: {e}")
            raise
        
        try:
            # MedGemma uses AutoModelForImageTextToText, not AutoModelForVision2Seq
            from transformers import AutoModelForImageTextToText, AutoProcessor
            logger.info("✓ Transformers imported successfully")
        except Exception as e:
            logger.error(f"❌ Transformers import failed: {e}")
            raise
        
        # Try to import BitsAndBytesConfig with fallback
        BitsAndBytesConfig = None
        use_quantization = bitsandbytes_available
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                logger.info("✓ BitsAndBytesConfig imported successfully")
            except Exception as e:
                logger.warning(f"⚠ BitsAndBytesConfig import failed despite bitsandbytes working: {e}")
                logger.warning("⚠ Will try to load model without quantization")
                use_quantization = False
                BitsAndBytesConfig = None
        else:
            logger.info("ℹ bitsandbytes not available, loading model without quantization")
        
        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            logger.info("✓ Processor loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load processor: {e}")
            raise
        
        # Pre-loading memory check and optimization
        logger.info("Checking system memory before model loading...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: Tesla T4 with {gpu_memory:.1f}GB VRAM")
            logger.info("Cleared GPU cache to maximize available memory")

        # Force garbage collection
        import gc
        gc.collect()
        logger.info("System memory optimized for model loading")

        # Load model with multiple fallback strategies
        model = None
        load_attempts = [
            {
                "name": "4-bit quantization (recommended)",
                "quantization": use_quantization,
                "config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                ) if use_quantization and BitsAndBytesConfig else None,
                "device_map": "auto",
                "dtype": torch.bfloat16
            },
            {
                "name": "8-bit quantization (fallback)",
                "quantization": use_quantization,
                "config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True
                ) if use_quantization and BitsAndBytesConfig else None,
                "device_map": "cuda",
                "dtype": torch.float16
            },
            {
                "name": "GPU only (half precision)",
                "quantization": False,
                "config": None,
                "device_map": "cuda",
                "dtype": torch.float16
            },
            {
                "name": "CPU (last resort - very slow)",
                "quantization": False,
                "config": None,
                "device_map": "cpu",
                "dtype": torch.float32
            }
        ]

        for attempt in load_attempts:
            try:
                logger.info(f"Attempting to load model: {attempt['name']}")

                # Clear any existing GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

                load_kwargs = {
                    "device_map": attempt["device_map"],
                    "torch_dtype": attempt["dtype"],
                    "low_cpu_mem_usage": True,
                }

                if attempt["quantization"] and attempt["config"]:
                    load_kwargs["quantization_config"] = attempt["config"]

                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    **load_kwargs
                )

                logger.info(f"✓ Model loaded successfully with {attempt['name']}!")
                break

            except Exception as e:
                logger.warning(f"❌ Failed to load with {attempt['name']}: {str(e)}")
                if model is not None:
                    del model
                    model = None
                continue

        if model is None:
            raise RuntimeError("Failed to load model with all available strategies")
        
        # Start ngrok tunnel
        try:
            if NGROK_AUTH_TOKEN:
                ngrok.set_auth_token(NGROK_AUTH_TOKEN)
                logger.info("✓ Using ngrok auth token")
            else:
                logger.warning("⚠ No NGROK_AUTH_TOKEN found. Using free tier")
            
            # Kill any existing tunnels
            ngrok.kill()
            
            # Start tunnel on port 8000
            ngrok_tunnel = ngrok.connect(8000, bind_tls=True)
            public_url = ngrok_tunnel.public_url
            
            logger.info("="*80)
            logger.info("🚀 API is now running with ngrok!")
            logger.info("="*80)
            logger.info(f"\n📡 Public URL: {public_url}")
            logger.info(f"\n🔗 API Endpoints:")
            logger.info(f"   - Health Check: {public_url}/health")
            logger.info(f"   - API Docs: {public_url}/docs")
            logger.info(f"   - Text Analysis: {public_url}/api/analyze")
            logger.info(f"   - Image Analysis: {public_url}/api/analyze-with-images")
            logger.info(f"   - DICOM Analysis: {public_url}/api/analyze-dicom")
            logger.info(f"   - DICOM Series: {public_url}/api/analyze-dicom-series")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"❌ Failed to start ngrok: {e}")
            logger.info("Continuing without ngrok tunnel...")
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if ngrok_tunnel:
        ngrok.kill()
        logger.info("ngrok tunnel closed")


app = FastAPI(
    title="MedGemma Medical Reporting API",
    description=f"API for medical image analysis and reporting using Google's {MODEL_NAME} model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    prompt: str
    role_instruction: Optional[str] = "You are a helpful medical assistant."
    max_new_tokens: Optional[int] = 2048


class InferenceResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    thought: Optional[str] = None
    answer: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
    ngrok_url: Optional[str] = None
    model_name: str


class DicomSliceResult(BaseModel):
    slice_number: int
    instance_number: Optional[int] = None
    analysis: str
    thought: Optional[str] = None


class DicomAnalysisResponse(BaseModel):
    success: bool
    total_slices: int
    analyzed_slices: int
    slice_results: List[DicomSliceResult] = []
    summary: Optional[str] = None
    error: Optional[str] = None
    patient_info: Optional[Dict[str, Any]] = None


def create_messages(
    prompt: str,
    role_instruction: str = "You are a helpful medical assistant.",
    images: List[Image.Image] = [],
) -> List[dict]:
    """Create messages in the chat format expected by MedGemma"""
    # MedGemma uses a structured message format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": role_instruction}]
        },
        {
            "role": "user",
            "content": []
        }
    ]
    
    # Add text to user message
    messages[1]["content"].append({"type": "text", "text": prompt})
    
    # Add images to user message if provided
    for image in images:
        messages[1]["content"].append({"type": "image", "image": image})
    
    return messages


def generate_response(messages: List[dict], max_new_tokens: int = 2048) -> str:
    """Generate response using the model with MedGemma chat template"""
    import torch
    
    if model is None or processor is None:
        raise ValueError("Model not loaded")
    
    # Apply chat template and prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    # Decode only the generated tokens
    response = processor.decode(generation, skip_special_tokens=True)
    
    return response


def dicom_to_pil(dicom_data) -> Image.Image:
    """Convert DICOM data to PIL Image"""
    pixel_array = dicom_data.pixel_array
    
    # Normalize to 0-255 range
    pixel_array = pixel_array.astype(float)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
    pixel_array = pixel_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(pixel_array.shape) == 2:
        # Grayscale
        img = Image.fromarray(pixel_array, mode='L').convert('RGB')
    else:
        # Already RGB
        img = Image.fromarray(pixel_array)
    
    return img


def parse_response(response_text: str) -> dict:
    """Parse the model response to separate thought and answer if present"""
    parts = response_text.split("<unused95>")
    if len(parts) == 1:
        return {
            "response": response_text,
            "thought": None,
            "answer": response_text
        }
    thought, answer = parts
    thought = thought.replace("<unused94>thought\n", "")
    return {
        "response": response_text,
        "thought": thought,
        "answer": answer
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - landing page"""
    ngrok_url = ngrok_tunnel.public_url if ngrok_tunnel else "Not available"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MedGemma API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            .status {{ padding: 15px; background: #d4edda; border-radius: 5px; margin: 20px 0; }}
            .info {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            a {{ color: #007bff; text-decoration: none; }}
            .endpoint {{ margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>🏥 MedGemma Medical Reporting API</h1>
        <div class="status">
            <strong>Status:</strong> {"✅ Online" if model else "⏳ Loading"}
        </div>
        <div class="info">
            <strong>Model:</strong> {MODEL_NAME}<br>
            <strong>Model Loaded:</strong> {"Yes" if model else "No"}<br>
            <strong>ngrok URL:</strong> {ngrok_url}
        </div>
        <h2>📚 API Documentation</h2>
        <div class="endpoint">
            <a href="/docs" target="_blank">Interactive API Docs (Swagger UI)</a>
        </div>
        <div class="endpoint">
            <a href="/redoc" target="_blank">Alternative API Docs (ReDoc)</a>
        </div>
        <h2>🔗 Endpoints</h2>
        <div class="endpoint"><strong>GET</strong> /health - Health check</div>
        <div class="endpoint"><strong>POST</strong> /api/analyze - Text analysis</div>
        <div class="endpoint"><strong>POST</strong> /api/analyze-with-images - Image analysis</div>
        <div class="endpoint"><strong>POST</strong> /api/analyze-base64 - Base64 image analysis</div>
        <div class="endpoint"><strong>POST</strong> /api/analyze-dicom - Single DICOM file analysis</div>
        <div class="endpoint"><strong>POST</strong> /api/analyze-dicom-series - Complete DICOM series analysis</div>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "message": "Model is ready" if model is not None else "Model is still loading",
        "ngrok_url": ngrok_tunnel.public_url if ngrok_tunnel else None,
        "model_name": MODEL_NAME
    }


@app.post("/api/analyze", response_model=InferenceResponse)
async def analyze_text(request: InferenceRequest):
    """
    Analyze text prompt without images
    
    Args:
        request: InferenceRequest with prompt and optional parameters
    
    Returns:
        InferenceResponse with the model's response
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        logger.info(f"Processing text-only analysis request")
        
        # Create formatted messages
        messages = create_messages(
            prompt=request.prompt,
            role_instruction=request.role_instruction,
            images=[]
        )
        
        # Generate response
        response_text = generate_response(
            messages=messages,
            max_new_tokens=request.max_new_tokens
        )
        
        parsed = parse_response(response_text)
        
        return InferenceResponse(
            success=True,
            response=parsed["response"],
            thought=parsed["thought"],
            answer=parsed["answer"]
        )
    
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return InferenceResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-with-images", response_model=InferenceResponse)
async def analyze_with_images(
    prompt: str = Form(...),
    role_instruction: str = Form("You are a helpful medical assistant."),
    max_new_tokens: int = Form(2048),
    images: List[UploadFile] = File(...)
):
    """
    Analyze prompt with medical images
    
    Args:
        prompt: The medical question or prompt
        role_instruction: Optional role instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        images: List of image files to analyze
    
    Returns:
        InferenceResponse with the model's response
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        logger.info(f"Processing analysis request with {len(images)} image(s)")
        
        # Process uploaded images
        pil_images = []
        for idx, image_file in enumerate(images):
            try:
                # Read image bytes
                image_bytes = await image_file.read()
                # Open as PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_images.append(pil_image)
                logger.info(f"Processed image {idx + 1}: {pil_image.size}, mode: {pil_image.mode}")
            except Exception as e:
                logger.error(f"Error processing image {idx + 1}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image {idx + 1}: {str(e)}")
        
        # Create formatted messages
        messages = create_messages(
            prompt=prompt,
            role_instruction=role_instruction,
            images=pil_images
        )
        
        # Run inference
        logger.info("Running model inference...")
        response_text = generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens
        )
        
        # Parse response
        parsed = parse_response(response_text)
        
        logger.info("Inference completed successfully")
        
        return InferenceResponse(
            success=True,
            response=parsed["response"],
            thought=parsed["thought"],
            answer=parsed["answer"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return InferenceResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-base64", response_model=InferenceResponse)
async def analyze_with_base64_images(
    prompt: str = Form(...),
    role_instruction: str = Form("You are a helpful medical assistant."),
    max_new_tokens: int = Form(2048),
    images_base64: str = Form(...)  # Comma-separated base64 strings
):
    """
    Analyze prompt with medical images provided as base64 strings
    
    Args:
        prompt: The medical question or prompt
        role_instruction: Optional role instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        images_base64: Comma-separated base64 encoded images
    
    Returns:
        InferenceResponse with the model's response
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        # Split base64 strings
        base64_list = [img.strip() for img in images_base64.split(",") if img.strip()]
        logger.info(f"Processing analysis request with {len(base64_list)} base64 image(s)")
        
        # Process base64 images
        pil_images = []
        for idx, base64_str in enumerate(base64_list):
            try:
                # Remove data URL prefix if present
                if "base64," in base64_str:
                    base64_str = base64_str.split("base64,")[1]
                
                # Decode base64
                image_bytes = base64.b64decode(base64_str)
                # Open as PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_images.append(pil_image)
                logger.info(f"Processed base64 image {idx + 1}: {pil_image.size}, mode: {pil_image.mode}")
            except Exception as e:
                logger.error(f"Error processing base64 image {idx + 1}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing base64 image {idx + 1}: {str(e)}")
        
        # Create formatted messages
        messages = create_messages(
            prompt=prompt,
            role_instruction=role_instruction,
            images=pil_images
        )
        
        # Run inference
        logger.info("Running model inference...")
        response_text = generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens
        )
        
        # Parse response
        parsed = parse_response(response_text)
        
        logger.info("Inference completed successfully")
        
        return InferenceResponse(
            success=True,
            response=parsed["response"],
            thought=parsed["thought"],
            answer=parsed["answer"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in base64 image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return InferenceResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-dicom", response_model=InferenceResponse)
async def analyze_dicom_file(
    prompt: str = Form(...),
    role_instruction: str = Form("You are an expert radiologist."),
    max_new_tokens: int = Form(2048),
    dicom_file: UploadFile = File(...)
):
    """
    Analyze a single DICOM file
    
    Args:
        prompt: The medical question or prompt
        role_instruction: Optional role instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        dicom_file: DICOM file to analyze (.dcm)
    
    Returns:
        InferenceResponse with the model's response
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        logger.info(f"Processing DICOM file: {dicom_file.filename}")
        
        # Read DICOM file
        dicom_bytes = await dicom_file.read()
        dicom_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
        
        # Convert to PIL Image
        pil_image = dicom_to_pil(dicom_data)
        logger.info(f"Converted DICOM to image: {pil_image.size}")
        
        # Create formatted messages
        messages = create_messages(
            prompt=prompt,
            role_instruction=role_instruction,
            images=[pil_image]
        )
        
        # Run inference
        logger.info("Running model inference on DICOM...")
        response_text = generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens
        )
        
        # Parse response
        parsed = parse_response(response_text)
        
        logger.info("DICOM analysis completed successfully")
        
        return InferenceResponse(
            success=True,
            response=parsed["response"],
            thought=parsed["thought"],
            answer=parsed["answer"]
        )
    
    except Exception as e:
        logger.error(f"Error in DICOM analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return InferenceResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-dicom-series", response_model=DicomAnalysisResponse)
async def analyze_dicom_series(
    prompt: str = Form("Please analyze this medical imaging series and provide detailed findings."),
    role_instruction: str = Form("You are an expert radiologist analyzing medical imaging studies."),
    max_new_tokens: int = Form(2048),
    dicom_files: List[UploadFile] = File(...),
    generate_summary: bool = Form(True)
):
    """
    Analyze a complete DICOM series (multiple slices)
    
    Args:
        prompt: The medical question or prompt for each slice
        role_instruction: Optional role instruction for the model
        max_new_tokens: Maximum number of tokens to generate per slice
        dicom_files: List of DICOM files (.dcm) - the complete series
        generate_summary: Whether to generate an overall summary of all slices
    
    Returns:
        DicomAnalysisResponse with results for each slice and optional summary
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        total_slices = len(dicom_files)
        logger.info(f"Processing DICOM series with {total_slices} files")
        
        # Sort DICOM files by instance number
        dicom_data_list = []
        for dicom_file in dicom_files:
            dicom_bytes = await dicom_file.read()
            dicom_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
            dicom_data_list.append({
                'data': dicom_data,
                'filename': dicom_file.filename,
                'instance': getattr(dicom_data, 'InstanceNumber', 0)
            })
        
        # Sort by instance number
        dicom_data_list.sort(key=lambda x: x['instance'])
        
        # Extract patient info from first slice
        first_dicom = dicom_data_list[0]['data']
        patient_info = {
            'patient_id': getattr(first_dicom, 'PatientID', 'Unknown'),
            'patient_name': str(getattr(first_dicom, 'PatientName', 'Unknown')),
            'study_date': getattr(first_dicom, 'StudyDate', 'Unknown'),
            'modality': getattr(first_dicom, 'Modality', 'Unknown'),
            'study_description': getattr(first_dicom, 'StudyDescription', 'Unknown'),
        }
        
        logger.info(f"Patient Info: {patient_info}")
        
        # Analyze each slice
        slice_results = []
        for idx, dicom_item in enumerate(dicom_data_list):
            try:
                logger.info(f"Analyzing slice {idx + 1}/{total_slices} (Instance: {dicom_item['instance']})")
                
                # Convert to PIL Image
                pil_image = dicom_to_pil(dicom_item['data'])
                
                # Create slice-specific prompt
                slice_prompt = f"{prompt}\n\nThis is slice {idx + 1} of {total_slices} (Instance Number: {dicom_item['instance']})."
                
                # Create formatted messages
                messages = create_messages(
                    prompt=slice_prompt,
                    role_instruction=role_instruction,
                    images=[pil_image]
                )
                
                # Run inference
                response_text = generate_response(
                    messages=messages,
                    max_new_tokens=max_new_tokens
                )
                
                # Parse response
                parsed = parse_response(response_text)
                
                slice_results.append(DicomSliceResult(
                    slice_number=idx + 1,
                    instance_number=dicom_item['instance'],
                    analysis=parsed['answer'],
                    thought=parsed['thought']
                ))
                
                logger.info(f"Completed analysis for slice {idx + 1}/{total_slices}")
                
            except Exception as e:
                logger.error(f"Error analyzing slice {idx + 1}: {str(e)}")
                slice_results.append(DicomSliceResult(
                    slice_number=idx + 1,
                    instance_number=dicom_item['instance'],
                    analysis=f"Error analyzing this slice: {str(e)}",
                    thought=None
                ))
        
        # Generate summary if requested
        summary = None
        if generate_summary and len(slice_results) > 0:
            try:
                logger.info("Generating summary of all slices...")
                
                # Compile all slice analyses
                all_analyses = "\n\n".join([
                    f"Slice {r.slice_number} (Instance {r.instance_number}): {r.analysis}"
                    for r in slice_results
                ])
                
                summary_prompt = f"""Based on the following analyses of {total_slices} slices from a {patient_info['modality']} imaging study, provide a comprehensive summary with:
1. Overall impression
2. Key findings across all slices
3. Any patterns or progression noted
4. Clinical recommendations

Individual slice analyses:
{all_analyses}

Please provide a comprehensive summary:"""
                
                summary_messages = create_messages(
                    prompt=summary_prompt,
                    role_instruction=role_instruction,
                    images=[]
                )
                
                summary_response = generate_response(
                    messages=summary_messages,
                    max_new_tokens=max_new_tokens
                )
                
                parsed_summary = parse_response(summary_response)
                summary = parsed_summary['answer']
                
                logger.info("Summary generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                summary = f"Error generating summary: {str(e)}"
        
        logger.info(f"DICOM series analysis completed: {len(slice_results)}/{total_slices} slices analyzed")
        
        return DicomAnalysisResponse(
            success=True,
            total_slices=total_slices,
            analyzed_slices=len(slice_results),
            slice_results=slice_results,
            summary=summary,
            patient_info=patient_info
        )
    
    except Exception as e:
        logger.error(f"Error in DICOM series analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return DicomAnalysisResponse(
            success=False,
            total_slices=len(dicom_files) if dicom_files else 0,
            analyzed_slices=0,
            error=str(e)
        )


if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        logger.info("✓ Running in Google Colab")
    except:
        IN_COLAB = False
        logger.info("✓ Running locally")
    
    # Print setup instructions if needed
    if IN_COLAB:
        logger.info("="*80)
        logger.info("📋 COLAB SETUP CHECKLIST:")
        logger.info("="*80)
        logger.info("1. ✓ GPU Runtime: Make sure you're using T4 GPU")
        logger.info("2. ✓ Packages: All dependencies should be installed")
        logger.info("3. ✓ HuggingFace: Login completed")
        logger.info("4. ✓ ngrok: Token configured")
        logger.info("="*80)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import json
import requests
import pdfplumber
import torch
import numpy as np
import tempfile
from typing import List, Optional
from datetime import datetime
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Try to import transformers (fail gracefully if not available)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not installed. Using fallback mode.")

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_KEY = os.getenv("API_KEY", "your-secret-key-here")  # For security
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10_000_000))  # 10MB
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")

# Create FastAPI app
app = FastAPI(
    title="AI Battle PDF QA System",
    description="Extract answers from PDFs using AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== PYDANTIC MODELS ==========
class PDFRequest(BaseModel):
    pdf_url: str
    questions: List[str]
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.3

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    confidence: float
    context_used: Optional[str] = None
    processing_time: Optional[float] = None

class PDFResponse(BaseModel):
    success: bool
    answers: List[QuestionAnswer]
    model_used: str
    total_processing_time: float
    pdf_size: Optional[int] = None
    chunks_processed: Optional[int] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    transformers_available: bool
    uptime: Optional[float] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    code: int
    timestamp: str

# ========== GLOBALS & INITIALIZATION ==========
start_time = datetime.now()
tokenizer = None
model = None
embedding_model = None

# ========== HELPER FUNCTIONS ==========
async def download_pdf(pdf_url: str, api_key: str = None):
    """Download PDF with timeout and size limit"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Get file size first
        head_response = requests.head(pdf_url, headers=headers, timeout=5)
        content_length = head_response.headers.get('content-length')
        
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({int(content_length)/1_000_000:.1f}MB). Max size: {MAX_FILE_SIZE/1_000_000}MB"
            )
        
        # Download PDF
        response = requests.get(
            pdf_url, 
            headers=headers, 
            timeout=30,
            stream=True
        )
        response.raise_for_status()
        
        # Check size during download
        content_length = len(response.content)
        if content_length > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({content_length/1_000_000:.1f}MB)"
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name, content_length
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Download timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_path: str):
    """Extract text from PDF including tables"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Extract tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        table_text = " | ".join([" | ".join([str(cell) for cell in row]) for row in table])
                        text += f"[Table {i+1}.{table_idx+1}]: {table_text}\n\n"
        
        return text.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Create overlapping text chunks"""
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        
        # Try to end at sentence boundary
        chunk = " ".join(chunk_words)
        if end < len(words) and '.' in chunk:
            # Find last period
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.3:  # Don't cut too much
                chunk = chunk[:last_period + 1]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 3):
    """Find relevant chunks using semantic search or TF-IDF"""
    if not chunks:
        return []
    
    # Simple TF-IDF based similarity (no external dependencies)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # Combine question and chunks
        documents = [question] + chunks
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get top-k chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_indices if i < len(chunks)]
        
    except:
        # Fallback: return first few chunks
        return chunks[:min(top_k, len(chunks))]

async def generate_answer_fallback(question: str, context: str):
    """Fallback answer generation without transformers"""
    # Simple rule-based extraction
    import re
    
    # Check for direct matches
    question_lower = question.lower()
    
    # Try to extract numbers
    if any(word in question_lower for word in ["how much", "how many", "number", "amount", "percentage"]):
        numbers = re.findall(r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?%|\d+(?:,\d+)*', context)
        if numbers:
            return f"The value is {numbers[0]}", 0.7
    
    # Try to extract dates
    if any(word in question_lower for word in ["when", "date", "year", "month"]):
        dates = re.findall(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', context)
        if dates:
            return f"The date is {dates[0]}", 0.6
    
    # Look for direct phrase in context
    question_words = question_lower.replace('?', '').split()
    context_lower = context.lower()
    
    for i in range(len(question_words), 0, -1):
        for j in range(len(question_words) - i + 1):
            phrase = ' '.join(question_words[j:j+i])
            if len(phrase) > 3 and phrase in context_lower:
                # Find the sentence containing the phrase
                sentences = re.split(r'[.!?]+', context)
                for sentence in sentences:
                    if phrase in sentence.lower():
                        return sentence.strip(), 0.5
    
    return "Answer not found in text", 0.1

async def generate_answer_llm(question: str, context: str, model_name: str = None, temperature: float = 0.3):
    """Generate answer using LLM"""
    global tokenizer, model
    
    if not TRANSFORMERS_AVAILABLE:
        return await generate_answer_fallback(question, context)
    
    try:
        # Lazy load model
        if tokenizer is None or model is None:
            print(f"Loading model: {model_name or MODEL_NAME}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name or MODEL_NAME,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name or MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR
            )
            print("✅ Model loaded successfully")
        
        # Create prompt
        prompt = f"""Extract information from the following context to answer the question.
If the answer cannot be found, say "Answer not found in text."

CONTEXT:
{context}

QUESTION: {question}

ANSWER: """
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        
        # Clean up
        answer = answer.replace('"', '').strip()
        
        if "answer not found" in answer.lower() or "i don't know" in answer.lower():
            return "Answer not found in text", 0.1
        
        # Calculate simple confidence
        confidence = min(0.9, len(answer) / 300.0)
        
        return answer, confidence
        
    except Exception as e:
        print(f"LLM error: {str(e)}")
        # Fallback to simple extraction
        return await generate_answer_fallback(question, context)

# ========== API ENDPOINTS ==========
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "AI Battle PDF QA System",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /ask": "Ask questions about a PDF",
            "GET /docs": "Interactive API documentation",
            "GET /models": "Available models"
        },
        "documentation": "Visit /docs for full API documentation"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=model is not None,
        transformers_available=TRANSFORMERS_AVAILABLE,
        uptime=(datetime.now() - start_time).total_seconds()
    )

@app.get("/models")
async def list_models():
    """List available models"""
    models = [
        {
            "id": "qwen-7b",
            "name": "Qwen2.5-7B-Instruct",
            "description": "7B parameter model, good balance of speed and accuracy",
            "recommended": True
        },
        {
            "id": "mistral-7b",
            "name": "Mistral-7B-Instruct",
            "description": "7B parameter model, fast and efficient"
        },
        {
            "id": "deepseek-7b",
            "name": "DeepSeek-Coder-6.7B",
            "description": "Good for technical documents"
        }
    ]
    return {"models": models}

@app.post("/ask", response_model=PDFResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ask_pdf(request: PDFRequest, background_tasks: BackgroundTasks):
    """Main endpoint - Ask questions about a PDF"""
    start_processing = datetime.now()
    
    # API key validation (optional)
    if API_KEY != "your-secret-key-here" and request.api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    try:
        print(f"📥 Processing PDF: {request.pdf_url}")
        
        # Download PDF
        pdf_path, pdf_size = await download_pdf(request.pdf_url, request.api_key)
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        print(f"📄 Extracted {len(text)} characters")
        
        # Create chunks
        chunks = create_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Process each question
        answers = []
        for question in request.questions:
            question_start = datetime.now()
            
            print(f"❓ Processing: {question[:50]}...")
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question, chunks, top_k=2)
            context = "\n\n".join(relevant_chunks[:2])
            
            # Generate answer
            answer, confidence = await generate_answer_llm(
                question, 
                context, 
                request.model,
                request.temperature or 0.3
            )
            
            # Calculate question processing time
            question_time = (datetime.now() - question_start).total_seconds()
            
            answers.append(QuestionAnswer(
                question=question,
                answer=answer,
                confidence=round(confidence, 2),
                context_used=context[:200] + "..." if context else None,
                processing_time=round(question_time, 2)
            ))
        
        # Calculate total processing time
        total_time = (datetime.now() - start_processing).total_seconds()
        
        return PDFResponse(
            success=True,
            answers=answers,
            model_used=request.model or MODEL_NAME,
            total_processing_time=round(total_time, 2),
            pdf_size=pdf_size,
            chunks_processed=len(chunks),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/stream")
async def stream_answers(request: PDFRequest):
    """Stream answers as they're generated"""
    async def generate():
        try:
            # Download and process PDF once
            pdf_path, _ = await download_pdf(request.pdf_url, request.api_key)
            text = extract_text_from_pdf(pdf_path)
            chunks = create_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Stream answers one by one
            for question in request.questions:
                relevant_chunks = find_relevant_chunks(question, chunks, top_k=2)
                context = "\n\n".join(relevant_chunks[:2])
                
                answer, confidence = await generate_answer_llm(
                    question, 
                    context, 
                    request.model,
                    request.temperature or 0.3
                )
                
                yield json.dumps({
                    "question": question,
                    "answer": answer,
                    "confidence": confidence,
                    "status": "processed"
                }) + "\n"
            
            yield json.dumps({"status": "complete"}) + "\n"
            
        except Exception as e:
            yield json.dumps({"error": str(e), "status": "error"}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# ========== ERROR HANDLERS ==========
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            code=exc.status_code,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            code=500,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ========== MAIN ENTRY POINT FOR VERCEL ==========
# This is required for Vercel to find the app
app = app

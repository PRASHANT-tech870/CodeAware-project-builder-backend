from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Import services
from services.project.project_service import ProjectService
from services.project.session_service import SessionService
from services.execution.execution_service import ExecutionService

# Import API routes
from routes.api import project, execution

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Google API key for Gemini
GOOGLE_API_KEY = "AIzaSyCMEPSK6GFQiZm48zO5dgE1QaoGYmcfQGw"

# Initialize services as singletons
session_service = SessionService()
execution_service = ExecutionService()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(project.router, prefix="/api", tags=["project"])
app.include_router(execution.router, prefix="/api", tags=["execution"])

# Define service accessor functions for dependency injection
def get_google_api_key():
    return GOOGLE_API_KEY

def get_session_service():
    return session_service

def get_execution_service():
    return execution_service

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "AI-Guided Project Builder API is running"}

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working correctly"}

# Import original endpoints to maintain compatibility during migration
# These will redirect to the modularized versions
from routes.api.project import start_project, get_next_step, get_step_questions
from routes.api.execution import execute_code, render_website, verify_step_completion, ask_question, terminate_streamlit

# Add aliases to maintain compatibility with existing frontend code
app.post("/start_project")(start_project)
app.post("/next_step")(get_next_step)
app.post("/get_step_questions")(get_step_questions)
app.post("/execute")(execute_code)
app.post("/render_website")(render_website)
app.post("/verify_step_completion")(verify_step_completion)
app.post("/ask_question")(ask_question)
app.post("/terminate_streamlit")(terminate_streamlit)

# For testing without Gemini API - uses the new modular structure
@app.post("/next_step_test")
async def next_step_test(request: dict):
    """
    Test endpoint for developing the step generation logic without using the Gemini API.
    Uses hardcoded steps that match the user's project type and idea.
    """
    logger.debug(f"Received next_step_test request: {request}")
    
    # Delegate to the project router version
    return await get_next_step(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
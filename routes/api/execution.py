from fastapi import APIRouter, HTTPException, Body
import logging
import traceback
from typing import Dict

from models.schemas import CodeRequest, StepCompletionRequest, QuestionRequest
from utils.helpers import generate_quiz_verification
from services.project.project_service import ProjectService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

def get_execution_service():
    from app import get_execution_service
    return get_execution_service()

def get_project_service():
    from app import get_google_api_key
    return ProjectService(get_google_api_key())

def get_session_service():
    from app import get_session_service
    return get_session_service()

@router.post("/execute")
async def execute_code(request: CodeRequest):
    """Execute code in the specified language"""
    execution_service = get_execution_service()
    
    try:
        result = execution_service.execute_code(request.code, request.language)
        return result
    except Exception as e:
        error_detail = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/render_website")
async def render_website(html_code: str = Body(..., embed=True), css_code: str = Body(..., embed=True)):
    """Render an HTML website with CSS"""
    try:
        execution_service = get_execution_service()
        logger.debug(f"Rendering website with HTML length {len(html_code)} and CSS length {len(css_code)}")
        
        result = execution_service.render_website(html_code, css_code)
        return result
    except Exception as e:
        error_detail = f"Error rendering website: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/verify_step_completion")
async def verify_step_completion(request: StepCompletionRequest):
    """Verify the user's understanding of a completed step using multiple choice questions"""
    try:
        # Get the user's answers and generate verification results
        verification_result = generate_quiz_verification(request.user_answers)
        
        return verification_result
    except Exception as e:
        error_detail = f"Error verifying step: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/ask_question")
async def ask_question(request: QuestionRequest):
    """Answer a user's question about code or a step"""
    try:
        # Get services
        project_service = get_project_service()
        session_service = get_session_service()
        
        # Check if session exists
        session = session_service.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
        
        # Prepare prompt for Gemini
        prompt = f"""
        I'm building a project using {request.project_type} with expertise level {session["expertise_level"]}.
        Project idea: {session["project_idea"]}
        
        Here is my current code:
        ```
        {request.code}
        ```
        
        My question is: {request.question}
        
        Please provide a helpful, detailed answer to my question considering my expertise level. Include code examples if relevant.
        """
        
        # Call Gemini API through project service
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # For questions, we don't need to process JSON, so return as is
        return {
            "response": response.text
        }
    except Exception as e:
        error_detail = f"Error generating answer: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/terminate_streamlit")
async def terminate_streamlit(request: Dict):
    """API endpoint to terminate a running Streamlit process"""
    execution_id = request.get("execution_id")
    if not execution_id:
        raise HTTPException(status_code=400, detail="Missing execution_id")
    
    execution_service = get_execution_service()
    success = execution_service.terminate_streamlit_process(execution_id)
    return {"success": success} 
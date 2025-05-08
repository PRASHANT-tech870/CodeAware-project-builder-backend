from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import tempfile
import os
import uuid
from pydantic import BaseModel, ValidationError
import base64
import google.generativeai as genai
from typing import Optional, List
import traceback
import logging
import json
import time
import signal
import psutil

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure API key for Google Gemini
GOOGLE_API_KEY = "AIzaSyCMEPSK6GFQiZm48zO5dgE1QaoGYmcfQGw"
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str
    language: str

class ProjectRequest(BaseModel):
    project_type: str  # "python+streamlit" or "html+css+js"
    expertise_level: str  # "beginner", "intermediate", "expert"
    project_idea: Optional[str] = None
    current_step: Optional[int] = 0

class StepRequest(BaseModel):
    project_type: str
    expertise_level: str
    project_idea: str
    current_step: int
    user_code: Optional[str] = None
    user_question: Optional[str] = None
    session_id: str
    user_understanding: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    code: str
    project_type: str
    session_id: str

class StepCompletionRequest(BaseModel):
    session_id: str
    step_number: int
    user_answers: List[dict]  # Changed from user_understanding to user_answers

# Store user sessions
user_sessions = {}

# Store user code files - indexed by session_id
user_code_files = {}

# Store running Streamlit processes keyed by execution_id
streamlit_processes = {}

# Add this helper function after the user_sessions declaration
def extract_json_from_response(text):
    """Extract JSON from a response that might be wrapped in markdown code blocks."""
    import re
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON inside markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, text)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find anything that looks like JSON object 
    json_object_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_object_pattern, text)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
    # Return the original text if we couldn't extract JSON
    return {"error": "Could not extract valid JSON", "original_text": text}

def terminate_streamlit_process(execution_id):
    """Terminate a Streamlit process by execution_id"""
    if execution_id in streamlit_processes:
        process = streamlit_processes[execution_id]
        if process.poll() is None:  # Process is still running
            try:
                # Try to terminate process group to ensure all child processes are killed
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Fallback to just terminating the process
                process.terminate()
                
            # Wait a bit for process to terminate
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
        
        # Remove from tracked processes
        del streamlit_processes[execution_id]
        return True
    return False

@app.post("/execute")
async def execute_code(request: CodeRequest):
    code = request.code
    language = request.language.lower()
    
    # Create a unique ID for this execution
    execution_id = str(uuid.uuid4())
    
    if language == "python":
        return execute_python(code, execution_id)
    elif language == "javascript":
        return execute_javascript(code, execution_id)
    elif language == "html":
        return render_html(code, execution_id)
    elif language == "css":
        return render_css(code, execution_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

def execute_python(code, execution_id):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file.write(code.encode())
        temp_file_path = temp_file.name
    
    try:
        # Check if this is a Streamlit app
        is_streamlit = 'import streamlit' in code or 'from streamlit' in code
        
        if is_streamlit:
            # Terminate any existing process with the same execution_id
            terminate_streamlit_process(execution_id)
            
            # Execute Streamlit with a timeout
            # We can't capture the web output, but we can report if it started successfully
            try:
                # Start Streamlit process on a random port
                port = 8501 + (hash(execution_id) % 100)  # Random port between 8501-8600
                
                # Build the command with appropriate flags to run headless
                streamlit_cmd = [
                    "streamlit", "run", 
                    temp_file_path,
                    "--server.address", "localhost",
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--server.runOnSave", "true",
                    "--browser.serverAddress", "localhost",
                    "--browser.gatherUsageStats", "false"
                ]
                
                # Use Popen instead of run so we don't block waiting for the process
                process = subprocess.Popen(
                    streamlit_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True  # This detaches the process so it continues running
                )
                
                # Store the process for later cleanup
                streamlit_processes[execution_id] = process
                
                # Wait a short time to see if the process starts up without errors
                time.sleep(2)
                
                # Check if the process is still running
                if process.poll() is None:
                    # Still running, assume success
                    return {
                        "stdout": "Streamlit app is running successfully at http://localhost:" + str(port),
                        "stderr": "",
                        "exit_code": 0,
                        "execution_id": execution_id,
                        "is_streamlit": True,
                        "streamlit_url": f"http://localhost:{port}",
                        "streamlit_port": port
                    }
                else:
                    # Process ended quickly, likely an error
                    stdout, stderr = process.communicate(timeout=1)
                    # Remove from tracked processes
                    if execution_id in streamlit_processes:
                        del streamlit_processes[execution_id]
                    return {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode,
                        "execution_id": execution_id,
                        "is_streamlit": True,
                        "streamlit_error": True
                    }
            
            except subprocess.TimeoutExpired:
                # Cleanup if process tracking was started
                if execution_id in streamlit_processes:
                    terminate_streamlit_process(execution_id)
                return {
                    "stdout": "",
                    "stderr": "Timeout while starting Streamlit app",
                    "exit_code": -1,
                    "execution_id": execution_id,
                    "is_streamlit": True,
                    "streamlit_error": True
                }
            except Exception as e:
                # Cleanup if process tracking was started
                if execution_id in streamlit_processes:
                    terminate_streamlit_process(execution_id)
                return {
                    "stdout": "",
                    "stderr": f"Error running Streamlit app: {str(e)}",
                    "exit_code": -1,
                    "execution_id": execution_id,
                    "is_streamlit": True,
                    "streamlit_error": True
                }
        else:
            # For regular Python, run normally
            result = subprocess.run(
                ["python3", temp_file_path],
                capture_output=True,
                text=True,
                timeout=10  # Set a timeout to prevent infinite loops
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "execution_id": execution_id,
                "is_streamlit": False
            }
    except subprocess.TimeoutExpired:
        # Cleanup if process tracking was started for Streamlit
        if is_streamlit and execution_id in streamlit_processes:
            terminate_streamlit_process(execution_id)
        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "exit_code": -1,
            "execution_id": execution_id,
            "is_streamlit": is_streamlit if 'is_streamlit' in locals() else False
        }
    finally:
        # Don't delete the file immediately as the Streamlit process might still be using it
        # We could implement a cleanup mechanism later
        if not is_streamlit:
            os.unlink(temp_file_path)

def execute_javascript(code, execution_id):
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as temp_file:
        temp_file.write(code.encode())
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(
            ["node", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "execution_id": execution_id
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "exit_code": -1,
            "execution_id": execution_id
        }
    finally:
        os.unlink(temp_file_path)

def render_html(html_code, execution_id):
    # For HTML, we'll just return the HTML code for the frontend to display
    # The frontend can then use an iframe or other method to show the rendered HTML
    return {
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "execution_id": execution_id,
        "html_content": html_code
    }

def render_css(css_code, execution_id):
    # For CSS, we'll return the CSS code for the frontend to use with HTML
    return {
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
        "execution_id": execution_id,
        "css_content": css_code
    }

@app.post("/render_website")
async def render_website(html_code: str = Body(..., embed=True), css_code: str = Body(..., embed=True)):
    # Create a temporary HTML file with the CSS embedded
    temp_dir = tempfile.mkdtemp()
    html_file_path = os.path.join(temp_dir, "index.html")
    
    # Combine HTML and CSS
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        {css_code}
        </style>
    </head>
    <body>
    {html_code}
    </body>
    </html>
    """
    
    with open(html_file_path, "w") as html_file:
        html_file.write(full_html)
    
    # Return the combined HTML
    return {
        "rendered_html": full_html,
        "status": "success"
    }

@app.post("/start_project")
async def start_project(request: ProjectRequest):
    # Generate a session ID for the user
    session_id = str(uuid.uuid4())
    
    logger.debug(f"Starting project with: project_type={request.project_type}, expertise_level={request.expertise_level}, project_idea={request.project_idea}")
    
    # Prepare prompt for Gemini based on project type and expertise level
    prompt = f"""
    I want to build a project using {request.project_type}. My expertise level is {request.expertise_level}.
    {f"Specifically, I want to build: {request.project_idea}" if request.project_idea else "Please suggest a good beginner-friendly project for me."}
    
    I need a project broken down into multiple small steps (levels) where each level completes a small part of the project, like a game with different levels.
    
    Please format the content of each step differently based on my expertise level:
    - beginner: Provide small code snippets with detailed explanation of each line
    - intermediate: Provide text instructions first with full steps of what to do (don't give code initially)
    - expert: Provide only workflow description, very minimal guidance
    
    Please provide:
    1. A brief introduction to the project
    2. A clear breakdown of steps to complete it (at least 8-10 different steps/levels)
    3. The first step with detailed explanation in the appropriate format for my expertise level
    
    Format your response as a JSON object with these fields:
    - project_title: A catchy name for the project
    - project_description: A brief overview of what we're building
    - total_steps: Total number of steps to complete the project
    - steps: An array of step objects, where each step has:
      - title: Step title
      - description: Detailed explanation formatted for my expertise level
      - code: Starter code for this step (for beginners), or empty for intermediate/expert
      - expected_outcome: What should happen after completing this step
      - quiz_question: A question that tests understanding of this step
      - quiz_answer: The correct answer or key points that should be in the answer
    
    For this request, just include the first step in the 'steps' array.
    
    IMPORTANT: Return only the raw JSON without any markdown formatting or code blocks. Do not include ```json or ``` around your response. Return only valid, parseable JSON.
    """
    
    try:
        logger.debug("Initializing Gemini model")
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )
        
        logger.debug("Sending prompt to Gemini API")
        logger.debug(f"Received response from Gemini API: {response}")
        
        # Process the response to extract clean JSON
        response_text = response.text
        
        # Try to extract JSON from the response
        try:
            # Extract JSON and convert back to string for storage
            json_data = extract_json_from_response(response_text)
            clean_response_text = json.dumps(json_data)
        except Exception as json_error:
            logger.error(f"Error extracting JSON from response: {str(json_error)}")
            clean_response_text = response_text
        
        # Store session data
        user_sessions[session_id] = {
            "project_type": request.project_type,
            "expertise_level": request.expertise_level,
            "project_idea": request.project_idea or "AI suggested project",
            "current_step": 0,
            "gemini_response": clean_response_text,
            "total_steps": json_data.get("total_steps", 10)  # Store total steps
        }
        
        # Initialize empty code files for this session
        user_code_files[session_id] = {
            "html": "",
            "css": "",
            "javascript": "",
            "python": ""
        }
        
        logger.debug(f"Session stored with ID: {session_id}")
        
        return {
            "session_id": session_id,
            "response": clean_response_text
        }
    except Exception as e:
        error_detail = f"Error generating project: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# Add a simplified version of start_project that uses static content for testing
@app.post("/start_project_test")
async def start_project_test(request: ProjectRequest):
    # Generate a session ID for the user
    session_id = str(uuid.uuid4())
    
    logger.debug(f"Starting test project with: project_type={request.project_type}, expertise_level={request.expertise_level}")
    
    # Create a static demo response based on project type
    if request.project_type == "html+css+js":
        demo_response = {
            "project_title": "Modern Portfolio Website",
            "project_description": "A responsive portfolio website to showcase your work and skills using HTML, CSS, and JavaScript.",
            "total_steps": 5,
            "steps": [
                {
                    "title": "Setting up the HTML structure",
                    "description": "In this step, we'll create the basic HTML structure for our portfolio website.",
                    "code": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>My Portfolio</title>\n</head>\n<body>\n  <header>\n    <h1>My Portfolio</h1>\n    <nav>\n      <ul>\n        <li><a href=\"#about\">About</a></li>\n        <li><a href=\"#projects\">Projects</a></li>\n        <li><a href=\"#contact\">Contact</a></li>\n      </ul>\n    </nav>\n  </header>\n  \n  <main>\n    <section id=\"about\">\n      <h2>About Me</h2>\n      <p>Welcome to my portfolio! I am a web developer passionate about creating beautiful and functional websites.</p>\n    </section>\n    \n    <section id=\"projects\">\n      <h2>My Projects</h2>\n      <div class=\"project\">\n        <h3>Project 1</h3>\n        <p>Description of project 1</p>\n      </div>\n      <div class=\"project\">\n        <h3>Project 2</h3>\n        <p>Description of project 2</p>\n      </div>\n    </section>\n    \n    <section id=\"contact\">\n      <h2>Contact Me</h2>\n      <form>\n        <input type=\"text\" placeholder=\"Name\">\n        <input type=\"email\" placeholder=\"Email\">\n        <textarea placeholder=\"Message\"></textarea>\n        <button type=\"submit\">Send</button>\n      </form>\n    </section>\n  </main>\n  \n  <footer>\n    <p>&copy; 2023 My Portfolio</p>\n  </footer>\n</body>\n</html>",
                    "expected_outcome": "A basic HTML structure for our portfolio website with header, navigation, main sections, and footer."
                }
            ]
        }
    else:  # python+streamlit
        demo_response = {
            "project_title": "Data Visualization Dashboard",
            "project_description": "A data visualization dashboard using Python and Streamlit to analyze and display interactive charts.",
            "total_steps": 5,
            "steps": [
                {
                    "title": "Setting up Streamlit app",
                    "description": "In this step, we'll create the basic structure for our Streamlit app and test that it runs correctly.",
                    "code": "import streamlit as st\nimport pandas as pd\nimport numpy as np\n\n# Set up the app title and description\nst.title('Data Visualization Dashboard')\nst.write('Welcome to my data visualization dashboard!')\n\n# Add a sidebar\nst.sidebar.header('Dashboard Controls')\n\n# Create some demo data\ndata = pd.DataFrame({\n    'Date': pd.date_range(start='2023-01-01', periods=10),\n    'Values': np.random.randn(10).cumsum()\n})\n\n# Display the data\nst.subheader('Sample Data')\nst.dataframe(data)\n\n# Create a simple chart\nst.subheader('Line Chart')\nst.line_chart(data.set_index('Date'))",
                    "expected_outcome": "A basic Streamlit app that displays a sample dataset and a line chart."
                }
            ]
        }
    
    # Convert to JSON string for storage
    response_text = json.dumps(demo_response)
    
    # Store session data
    user_sessions[session_id] = {
        "project_type": request.project_type,
        "expertise_level": request.expertise_level,
        "project_idea": request.project_idea or "AI suggested project",
        "current_step": 0,
        "gemini_response": response_text,
        "total_steps": demo_response.get("total_steps", 5)  # Store total steps
    }
    
    # Initialize empty code files for this session
    user_code_files[session_id] = {
        "html": "",
        "css": "",
        "javascript": "",
        "python": ""
    }
    
    return {
        "session_id": session_id,
        "response": response_text
    }

@app.post("/next_step")
async def get_next_step(request: dict):
    # Log the incoming request for debugging
    logger.debug(f"Received next_step request: {request}")
    
    try:
        # Convert the dict to a StepRequest model
        step_request = StepRequest(**request)
        session_id = step_request.session_id
        
        # Check if session exists
        if session_id not in user_sessions:
            error_msg = f"Session not found: {session_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = user_sessions[session_id]
        next_step = step_request.current_step + 1
        
        # Check if we've reached the total number of steps
        total_steps = session.get("total_steps", 10)  # Default to 10 if not specified
        if next_step > total_steps:
            return {
                "session_id": session_id,
                "response": json.dumps({
                    "project_completed": True,
                    "message": "Congratulations! You have completed all steps in this project."
                })
            }
        
        logger.debug(f"Processing next step {next_step} for session {session_id}")
        
        # Save the user's code for this step
        if session_id not in user_code_files:
            user_code_files[session_id] = {
                "html": "",
                "css": "",
                "javascript": "",
                "python": ""
            }
        
        # Get current user code 
        current_code_type = None
        if step_request.user_code:
            # Try to determine code type
            if "<html" in step_request.user_code or "<body" in step_request.user_code:
                current_code_type = "html"
                user_code_files[session_id]["html"] = step_request.user_code
            elif "{" in step_request.user_code and "font" in step_request.user_code:
                current_code_type = "css"
                user_code_files[session_id]["css"] = step_request.user_code
            elif "function" in step_request.user_code or "var " in step_request.user_code or "const " in step_request.user_code:
                current_code_type = "javascript"
                user_code_files[session_id]["javascript"] = step_request.user_code
            elif "import" in step_request.user_code or "def " in step_request.user_code:
                current_code_type = "python"
                user_code_files[session_id]["python"] = step_request.user_code
            else:
                # Default to the language from the project type
                if step_request.project_type == "html+css+js":
                    current_code_type = "html"
                    user_code_files[session_id]["html"] = step_request.user_code
                else:
                    current_code_type = "python"
                    user_code_files[session_id]["python"] = step_request.user_code
    
        # Prepare prompt for Gemini based on the current step and user's code
        prompt = f"""
        I'm building a project using {step_request.project_type} with expertise level {step_request.expertise_level}.
        Project idea: {step_request.project_idea}
        
        I have completed step {step_request.current_step} and here is my code:
        ```
        {step_request.user_code or "No code provided"}
        ```
        
        {f"I have a question: {step_request.user_question}" if step_request.user_question else ""}
        
        {f"This is my understanding of the step: {step_request.user_understanding}" if step_request.user_understanding else ""}
        
        Please provide step {next_step} with:
        1. Detailed feedback on my current code - be specific about what I did well and what could be improved
        2. Detailed explanation of the next step, formatted based on my expertise level:
          - beginner: Provide small code snippets with detailed explanation of each line
          - intermediate: Provide text instructions first with full steps of what to do (don't give code initially)
          - expert: Provide only workflow description, very minimal guidance
        3. Starter code for the next step (if I'm a beginner)
        
        Format your response as a JSON object with these fields:
        - step_number: The step number
        - title: Step title
        - feedback: Feedback on the user's current code
        - description: Detailed explanation of this step
        - code: Starter code for this step (detailed for beginners, minimal or empty for others)
        - expected_outcome: What should happen after completing this step
        - quiz_question: A question that tests understanding of this step
        - quiz_answer: The correct answer or key points that should be in the answer
        
        IMPORTANT: Return only the raw JSON without any markdown formatting or code blocks. Do not include ```json or ``` around your response. Return only valid, parseable JSON.
        """
        
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )
        
        # Process the response to extract clean JSON
        response_text = response.text
        logger.debug(f"Raw Gemini response: {response_text[:200]}...")
        
        # Try to extract JSON from the response
        try:
            # Extract JSON and convert back to string for storage
            json_data = extract_json_from_response(response_text)
            clean_response_text = json.dumps(json_data)
            logger.debug(f"Extracted JSON: {clean_response_text[:200]}...")
        except Exception as json_error:
            logger.error(f"Error extracting JSON from response: {str(json_error)}")
            clean_response_text = response_text
        
        # Update session data
        session["current_step"] = next_step
        
        return {
            "session_id": session_id,
            "response": clean_response_text
        }
    except ValidationError as ve:
        error_detail = f"Validation error: {str(ve)}"
        logger.error(error_detail)
        raise HTTPException(status_code=422, detail=error_detail)
    except Exception as e:
        error_detail = f"Error generating next step: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/ask_question")
async def ask_question(request: dict):
    # Log the incoming request for debugging
    logger.debug(f"Received ask_question request: {request}")
    
    try:
        # Convert the dict to a QuestionRequest model
        question_request = QuestionRequest(**request)
        
        # Check if session exists
        if question_request.session_id not in user_sessions:
            error_msg = f"Session not found: {question_request.session_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = user_sessions[question_request.session_id]
        
        # Prepare prompt for Gemini
        prompt = f"""
        I'm building a project using {question_request.project_type} with expertise level {session["expertise_level"]}.
        Project idea: {session["project_idea"]}
        
        Here is my current code:
        ```
        {question_request.code}
        ```
        
        My question is: {question_request.question}
        
        Please provide a helpful, detailed answer to my question considering my expertise level. Include code examples if relevant.
        """
        
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # For questions, we don't need to process JSON, so return as is
        return {
            "response": response.text
        }
    except ValidationError as ve:
        error_detail = f"Validation error: {str(ve)}"
        logger.error(error_detail)
        raise HTTPException(status_code=422, detail=error_detail)
    except Exception as e:
        error_detail = f"Error generating answer: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/verify_step_completion")
async def verify_step_completion(request: StepCompletionRequest):
    """Verify the user's understanding of a completed step using multiple choice questions"""
    logger.debug(f"Received step verification: {request}")
    
    try:
        session_id = request.session_id
        
        # Check if session exists
        if session_id not in user_sessions:
            error_msg = f"Session not found: {session_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = user_sessions[session_id]
        
        # For debugging
        logger.debug(f"Session data for verification: {session}")
        
        # Get current step data from project steps
        if 'gemini_response' not in session:
            # If no previous steps are found, initialize them
            session['gemini_response'] = json.dumps({
                "steps": []
            })
        
        # Parse the gemini response to get the expected answer
        response_object = json.loads(session['gemini_response'])
        current_step = request.step_number
        
        # Evaluate user's answers
        correct_answers = 0
        total_questions = len(request.user_answers)
        
        # This will store feedback for each question
        question_feedback = []
        
        # Check each answer
        for i, answer in enumerate(request.user_answers):
            question_id = answer.get('question_id')
            user_answer = answer.get('answer')
            correct_answer = answer.get('correct_answer')
            
            # Basic validation
            if user_answer is None or correct_answer is None:
                question_feedback.append({
                    "question_id": question_id,
                    "correct": False,
                    "feedback": "Invalid answer format"
                })
                continue
            
            # Check if answer is correct (case-insensitive string comparison)
            is_correct = False
            if isinstance(user_answer, str) and isinstance(correct_answer, str):
                is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
            else:
                is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_answers += 1
                question_feedback.append({
                    "question_id": question_id,
                    "correct": True,
                    "feedback": "Correct answer!"
                })
            else:
                question_feedback.append({
                    "question_id": question_id,
                    "correct": False,
                    "feedback": f"Incorrect. The correct answer is: {correct_answer}"
                })
        
        # Calculate score
        score = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0
        
        # Determine if user passed (require all answers to be correct)
        passed = correct_answers == total_questions
        
        return {
            "correct": passed,
            "score": score,
            "feedback": "All answers correct!" if passed else "Some answers were incorrect. Please try again.",
            "question_feedback": question_feedback
        }
            
    except Exception as e:
        error_detail = f"Error verifying step: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/get_step_questions")
async def get_step_questions(request: dict):
    """Generate quiz questions for the current step"""
    try:
        session_id = request.get("session_id")
        step_number = request.get("step_number")
        
        # Check if session exists
        if session_id not in user_sessions:
            error_msg = f"Session not found: {session_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = user_sessions[session_id]
        
        # Parse the gemini response
        if 'gemini_response' not in session:
            error_msg = "Session data not found"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        response_object = json.loads(session['gemini_response'])
        
        # Use Gemini to generate questions about this step
        prompt = f"""
        Generate 2-3 multiple choice questions about the following step in a coding project.
        
        The step is about: {response_object.get('project_title', 'Web development')}
        Step number: {step_number + 1}
        
        These questions should:
        1. Test understanding of key concepts from this specific step ONLY
        2. Be simple and straightforward
        3. Each have 3-4 possible answers with only one correct answer
        4. Cover different aspects of what was taught in this step
        
        Respond with a JSON array of question objects, each containing:
        {{
            "question_id": "q1", // unique ID for the question
            "question_text": "the question text",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "the correct option exactly as written in options"
        }}
        
        Only return the JSON array with no additional text.
        """
        
        # Call Gemini for generating questions
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )
            
            # Process the response
            json_data = extract_json_from_response(response.text)
            
            # If we didn't get an array, wrap it
            if not isinstance(json_data, list):
                if isinstance(json_data, dict):
                    json_data = [json_data]
                else:
                    # Fallback questions if parsing failed
                    json_data = [
                        {
                            "question_id": "q1",
                            "question_text": "What is the main topic of this step?",
                            "options": ["HTML basics", "CSS styling", "JavaScript functions", "Python coding"],
                            "correct_answer": "HTML basics"  # Default, will likely be wrong
                        },
                        {
                            "question_id": "q2",
                            "question_text": "What is the expected outcome of this step?",
                            "options": ["A styled webpage", "A functional form", "Basic HTML structure", "Running Python code"],
                            "correct_answer": "Basic HTML structure"  # Default, will likely be wrong
                        }
                    ]
            
            return {
                "questions": json_data
            }
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            # Fallback with simple questions
            return {
                "questions": [
                    {
                        "question_id": "q1",
                        "question_text": "What is the main topic of this step?",
                        "options": ["HTML basics", "CSS styling", "JavaScript functions", "Python coding"],
                        "correct_answer": "HTML basics"  # Default, will likely be wrong
                    },
                    {
                        "question_id": "q2",
                        "question_text": "What is the expected outcome of this step?",
                        "options": ["A styled webpage", "A functional form", "Basic HTML structure", "Running Python code"],
                        "correct_answer": "Basic HTML structure"  # Default, will likely be wrong
                    }
                ]
            }
        
    except Exception as e:
        error_detail = f"Error getting questions: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/")
async def root():
    return {"message": "AI-Guided Project Builder API is running"}

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working correctly"}

# Add a test endpoint for next_step that doesn't use Gemini API
@app.post("/next_step_test")
async def next_step_test(request: dict):
    logger.debug(f"Received next_step_test request: {request}")
    
    try:
        # Convert the dict to a StepRequest model
        step_request = StepRequest(**request)
        session_id = step_request.session_id
        
        # Check if session exists
        if session_id not in user_sessions:
            error_msg = f"Session not found: {session_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = user_sessions[session_id]
        next_step = step_request.current_step + 1
        
        # Check if we've reached the total number of steps
        total_steps = session.get("total_steps", 5)  # Default to 5 for testing
        if next_step > total_steps:
            return {
                "session_id": session_id,
                "response": json.dumps({
                    "project_completed": True,
                    "message": "Congratulations! You have completed all steps in this project."
                })
            }
        
        # Save the user's code for this step
        if session_id not in user_code_files:
            user_code_files[session_id] = {
                "html": "",
                "css": "",
                "javascript": "",
                "python": ""
            }
        
        # Get current user code 
        current_code_type = None
        if step_request.user_code:
            # Try to determine code type
            if "<html" in step_request.user_code or "<body" in step_request.user_code:
                current_code_type = "html"
                user_code_files[session_id]["html"] = step_request.user_code
            elif "{" in step_request.user_code and "font" in step_request.user_code:
                current_code_type = "css"
                user_code_files[session_id]["css"] = step_request.user_code
            elif "function" in step_request.user_code or "var " in step_request.user_code or "const " in step_request.user_code:
                current_code_type = "javascript"
                user_code_files[session_id]["javascript"] = step_request.user_code
            elif "import" in step_request.user_code or "def " in step_request.user_code:
                current_code_type = "python"
                user_code_files[session_id]["python"] = step_request.user_code
            else:
                # Default to the language from the project type
                if step_request.project_type == "html+css+js":
                    current_code_type = "html"
                    user_code_files[session_id]["html"] = step_request.user_code
                else:
                    current_code_type = "python"
                    user_code_files[session_id]["python"] = step_request.user_code
        
        # Create project-specific sequential steps
        
        # Web Development Project Flow (HTML/CSS/JS) - Portfolio Website
        html_css_js_portfolio_steps = {
            1: {
                "title": "HTML Structure",
                "description": "Let's create the basic HTML structure for our portfolio website. We'll set up the document with proper HTML5 tags, create a header with navigation, and add sections for about, projects, and contact information.",
                "language": "html",
                "code": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>My Portfolio</title>\n</head>\n<body>\n  <header>\n    <h1>My Portfolio</h1>\n    <nav>\n      <ul>\n        <li><a href=\"#about\">About</a></li>\n        <li><a href=\"#projects\">Projects</a></li>\n        <li><a href=\"#contact\">Contact</a></li>\n      </ul>\n    </nav>\n  </header>\n  \n  <main>\n    <section id=\"about\">\n      <h2>About Me</h2>\n      <p>Welcome to my portfolio! I am a web developer passionate about creating beautiful and functional websites.</p>\n    </section>\n    \n    <section id=\"projects\">\n      <h2>My Projects</h2>\n      <div class=\"project\">\n        <h3>Project 1</h3>\n        <p>Description of project 1</p>\n      </div>\n      <div class=\"project\">\n        <h3>Project 2</h3>\n        <p>Description of project 2</p>\n      </div>\n    </section>\n    \n    <section id=\"contact\">\n      <h2>Contact Me</h2>\n      <form>\n        <label for=\"name\">Name:</label>\n        <input type=\"text\" id=\"name\" name=\"name\"><br>\n        \n        <label for=\"email\">Email:</label>\n        <input type=\"email\" id=\"email\" name=\"email\"><br>\n        \n        <label for=\"message\">Message:</label>\n        <textarea id=\"message\" name=\"message\"></textarea><br>\n        \n        <button type=\"submit\">Submit</button>\n      </form>\n    </section>\n  </main>\n  \n  <footer>\n    <p>&copy; 2023 My Portfolio. All rights reserved.</p>\n  </footer>\n</body>\n</html>",
                "expected_outcome": "A basic HTML structure for our portfolio website with header, navigation, main sections, and footer."
            },
            2: {
                "title": "Basic CSS Styling",
                "description": "Now let's add basic CSS styling to our portfolio. We'll create a stylesheet to set fonts, colors, and layout for the main elements like header, navigation, and sections.",
                "language": "css",
                "code": "/* General styles */\nbody {\n  font-family: Arial, sans-serif;\n  line-height: 1.6;\n  margin: 0;\n  padding: 0;\n  color: #333;\n}\n\nheader {\n  background-color: #4a89dc;\n  color: white;\n  padding: 1rem;\n  text-align: center;\n}\n\nnav ul {\n  display: flex;\n  justify-content: center;\n  list-style: none;\n  padding: 0;\n}\n\nnav ul li {\n  margin: 0 15px;\n}\n\nnav a {\n  color: white;\n  text-decoration: none;\n}\n\nmain {\n  max-width: 1200px;\n  margin: 0 auto;\n  padding: 20px;\n}\n\nsection {\n  margin-bottom: 40px;\n}\n\nfooter {\n  background-color: #333;\n  color: white;\n  text-align: center;\n  padding: 1rem;\n  margin-top: 2rem;\n}",
                "expected_outcome": "A styled portfolio website with improved visual appearance. The header will have a blue background, navigation will be centered, and the content will be properly spaced."
            },
            3: {
                "title": "Contact Form Styling",
                "description": "Let's enhance our portfolio by styling the contact form. We'll add proper spacing, borders, and styling to input elements and button to make the form more attractive and user-friendly.",
                "language": "css",
                "code": "",
                "expected_outcome": "A professional-looking contact form with styled inputs, proper spacing, and an attractive submit button."
            },
            4: {
                "title": "Project Section Enhancements",
                "description": "In this step, we'll improve the Projects section by adding proper grid layout, cards for each project, and styling that showcases your work effectively.",
                "language": "css",
                "code": "",
                "expected_outcome": "An attractive projects section with a grid layout showcasing each project in a card format."
            },
            5: {
                "title": "Form Validation with JavaScript",
                "description": "Now we'll add form validation to our contact form using JavaScript. We'll ensure that all fields are properly filled out before submission and provide feedback to the user about any errors.",
                "language": "javascript",
                "code": "",
                "expected_outcome": "A contact form with client-side validation that checks all fields before submission and provides user feedback."
            }
        }
        
        # Web Development Project Flow (HTML/CSS/JS) - Todo App
        html_css_js_todo_steps = {
            1: {
                "title": "HTML Structure for Todo App",
                "description": "Let's create the basic HTML structure for our Todo application. We'll set up the document with a form to add new tasks and a list to display existing tasks.",
                "language": "html",
                "code": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>Todo App</title>\n</head>\n<body>\n  <div class=\"container\">\n    <h1>My Todo List</h1>\n    \n    <form id=\"todo-form\">\n      <input type=\"text\" id=\"todo-input\" placeholder=\"Add a new task...\" required>\n      <button type=\"submit\">Add Task</button>\n    </form>\n    \n    <ul id=\"todo-list\">\n      <!-- Tasks will be added here dynamically -->\n      <li class=\"todo-item\">\n        <span class=\"todo-text\">Example task 1</span>\n        <button class=\"delete-btn\">Delete</button>\n      </li>\n      <li class=\"todo-item\">\n        <span class=\"todo-text\">Example task 2</span>\n        <button class=\"delete-btn\">Delete</button>\n      </li>\n    </ul>\n  </div>\n</body>\n</html>",
                "expected_outcome": "A basic HTML structure for our Todo app with input form and task list."
            },
            2: {
                "title": "CSS Styling for Todo App",
                "description": "Now let's style our Todo app to make it more visually appealing. We'll add colors, spacing, and effects to make the app look professional.",
                "language": "css",
                "code": "* {\n  margin: 0;\n  padding: 0;\n  box-sizing: border-box;\n}\n\nbody {\n  font-family: 'Arial', sans-serif;\n  background-color: #f5f5f5;\n  color: #333;\n  line-height: 1.6;\n}\n\n.container {\n  max-width: 500px;\n  margin: 50px auto;\n  padding: 20px;\n  background-color: white;\n  border-radius: 5px;\n  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);\n}\n\nh1 {\n  text-align: center;\n  margin-bottom: 20px;\n  color: #2c3e50;\n}\n\n#todo-form {\n  display: flex;\n  margin-bottom: 20px;\n}\n\n#todo-input {\n  flex: 1;\n  padding: 10px;\n  border: 1px solid #ddd;\n  border-radius: 4px 0 0 4px;\n  font-size: 16px;\n}\n\n#todo-form button {\n  padding: 10px 15px;\n  background-color: #3498db;\n  color: white;\n  border: none;\n  border-radius: 0 4px 4px 0;\n  cursor: pointer;\n  font-size: 16px;\n}\n\n#todo-form button:hover {\n  background-color: #2980b9;\n}\n\n#todo-list {\n  list-style: none;\n}\n\n.todo-item {\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  padding: 10px;\n  border-bottom: 1px solid #eee;\n}\n\n.todo-text {\n  flex: 1;\n}\n\n.delete-btn {\n  padding: 5px 10px;\n  background-color: #e74c3c;\n  color: white;\n  border: none;\n  border-radius: 4px;\n  cursor: pointer;\n}\n\n.delete-btn:hover {\n  background-color: #c0392b;\n}",
                "expected_outcome": "A styled Todo app with proper colors, spacing, and visual hierarchy."
            },
            3: {
                "title": "Add Task Functionality",
                "description": "In this step, we'll implement the JavaScript functionality to add new tasks to our Todo list. We'll create event listeners for the form submission and dynamically add new tasks to the list.",
                "language": "javascript",
                "code": "",
                "expected_outcome": "A functional Todo app where users can add new tasks that get displayed in the list."
            },
            4: {
                "title": "Delete and Complete Tasks",
                "description": "Now let's add functionality to delete tasks and mark them as completed. We'll implement event delegation for the delete buttons and add a way to toggle task completion status.",
                "language": "javascript",
                "code": "",
                "expected_outcome": "A Todo app where users can delete tasks and mark them as completed (with strikethrough styling)."
            },
            5: {
                "title": "Local Storage Integration",
                "description": "In this final step, we'll make our Todo app persistent by saving tasks to localStorage. This ensures that tasks remain even when the user refreshes the page or comes back later.",
                "language": "javascript",
                "code": "",
                "expected_outcome": "A complete Todo app that persists data between page refreshes using localStorage."
            }
        }
        
        # Python+Streamlit Data Dashboard Project
        python_streamlit_dashboard_steps = {
            1: {
                "title": "Basic Streamlit App Setup",
                "description": "Let's set up a basic Streamlit app for our data dashboard. We'll import the necessary libraries, create a title and description, and display some sample data.",
                "language": "python",
                "code": "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Set up the app title and description\nst.title('Data Visualization Dashboard')\nst.write('Welcome to my interactive data dashboard!')\n\n# Create some sample data\ndata = pd.DataFrame({\n    'Date': pd.date_range(start='2023-01-01', periods=10),\n    'Sales': np.random.randint(100, 500, size=10),\n    'Customers': np.random.randint(10, 50, size=10),\n    'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 10)\n})\n\n# Display the data\nst.subheader('Sample Data')\nst.dataframe(data)\n\n# Create a simple chart\nst.subheader('Sales Over Time')\nst.line_chart(data.set_index('Date')['Sales'])",
                "expected_outcome": "A basic Streamlit app that displays a sample dataset and a line chart of sales over time."
            },
            2: {
                "title": "Adding Sidebar Controls",
                "description": "Now let's add interactive controls to our dashboard using Streamlit's sidebar functionality. We'll create filters for date range and category selection.",
                "language": "python",
                "code": "",
                "expected_outcome": "A dashboard with a sidebar containing filters that let users control what data is displayed."
            },
            3: {
                "title": "Multiple Chart Types",
                "description": "In this step, we'll enhance our dashboard by adding different types of visualizations. We'll implement bar charts, scatter plots, and pie charts to represent different aspects of our data.",
                "language": "python",
                "code": "",
                "expected_outcome": "A dashboard with multiple visualization types showing different perspectives on the data."
            },
            4: {
                "title": "Data Analysis Features",
                "description": "Let's add some data analysis capabilities to our dashboard. We'll calculate and display summary statistics, trends, and insights from the data.",
                "language": "python",
                "code": "",
                "expected_outcome": "A dashboard that not only visualizes data but also provides analytical insights."
            },
            5: {
                "title": "File Upload and Custom Data",
                "description": "In our final step, we'll allow users to upload their own CSV files to analyze. We'll add file upload functionality and adapt our visualizations to work with custom data.",
                "language": "python",
                "code": "",
                "expected_outcome": "A complete dashboard that can analyze and visualize user-uploaded data files."
            }
        }
        
        # Python+Streamlit Machine Learning App
        python_streamlit_ml_steps = {
            1: {
                "title": "Basic ML App Setup",
                "description": "Let's create a basic Streamlit app for our machine learning project. We'll set up the structure and import necessary libraries for a simple predictive model.",
                "language": "python",
                "code": "import streamlit as st\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\n# Set up the app title and description\nst.title('Machine Learning Predictor')\nst.write('This app demonstrates a simple machine learning model using the Iris dataset.')\n\n# Load the Iris dataset\n@st.cache_data\ndef load_data():\n    from sklearn.datasets import load_iris\n    iris = load_iris()\n    df = pd.DataFrame(iris.data, columns=iris.feature_names)\n    df['target'] = iris.target\n    df['target_names'] = df['target'].apply(lambda x: iris.target_names[x])\n    return df\n\ndf = load_data()\n\n# Display the dataset\nst.subheader('Iris Dataset')\nst.dataframe(df.head())\n\n# Display basic statistics\nst.subheader('Dataset Statistics')\nst.write(df.describe())",
                "expected_outcome": "A basic Streamlit app that loads and displays the Iris dataset, ready for machine learning modeling."
            },
            2: {
                "title": "Training a Basic Model",
                "description": "In this step, we'll implement a simple machine learning model using scikit-learn. We'll train a Random Forest classifier on the Iris dataset and display the model's accuracy.",
                "language": "python",
                "code": "",
                "expected_outcome": "A Streamlit app that trains a machine learning model and displays its accuracy."
            },
            3: {
                "title": "Interactive Predictions",
                "description": "Now let's add interactive prediction capabilities. We'll create input fields for users to enter feature values and see real-time predictions from our model.",
                "language": "python",
                "code": "",
                "expected_outcome": "An app where users can enter values and get predictions from the trained model."
            },
            4: {
                "title": "Model Parameter Tuning",
                "description": "In this step, we'll add controls for users to adjust the machine learning model's parameters. We'll let them experiment with hyperparameters and see how they affect model performance.",
                "language": "python",
                "code": "",
                "expected_outcome": "An app that allows users to tune model parameters and observe changes in performance."
            },
            5: {
                "title": "Visualizing Model Results",
                "description": "Finally, we'll add visualizations to help users understand the model's performance. We'll create confusion matrices, feature importance plots, and prediction probability charts.",
                "language": "python",
                "code": "",
                "expected_outcome": "A complete machine learning app with visualizations that help interpret the model's predictions and performance."
            }
        }
        
        # Determine which project steps to use based on project idea keywords or default based on project type
        project_idea = step_request.project_idea.lower() if step_request.project_idea else ""
        
        if step_request.project_type == "html+css+js":
            if "todo" in project_idea or "task" in project_idea or "list" in project_idea:
                steps = html_css_js_todo_steps
            else:
                # Default to portfolio
                steps = html_css_js_portfolio_steps
        else:  # python+streamlit
            if "machine" in project_idea or "ml" in project_idea or "predict" in project_idea or "model" in project_idea:
                steps = python_streamlit_ml_steps
            else:
                # Default to data dashboard
                steps = python_streamlit_dashboard_steps
        
        # Make sure we have this step
        if next_step not in steps:
            next_step = min(next_step, len(steps))  # Cap at max steps available
            
        step_data = steps[next_step].copy()
        
        # Add step number explicitly
        step_data["step_number"] = next_step
        
        # Format title with correct step number
        step_data["title"] = f"Step {next_step}: {step_data['title']}"
        
        # Add appropriate feedback based on expertise level
        if step_request.expertise_level == "beginner":
            feedback = "Here's the next step. I'll provide some guidance to help you get started."
        elif step_request.expertise_level == "intermediate":
            feedback = "Great job! For this step, try implementing the solution using the description before looking at any code examples."
            # Remove code for intermediate level if not first step
            if next_step > 1:
                step_data["code"] = ""
        else:  # expert
            feedback = "Ready for the next challenge? Implement this step using your expertise."
            # Always remove code for expert level
            step_data["code"] = ""
            
        step_data["feedback"] = feedback
        
        # Generate appropriate quiz questions for this step based on the specific step content
        if step_request.project_type == "html+css+js":
            if step_data["language"] == "html":
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "What HTML tag is used for the main heading on a page?",
                        "options": ["<p>", "<h1>", "<header>", "<title>"],
                        "correct_answer": "<h1>"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "Which HTML element is used to create a list of items?",
                        "options": ["<div>", "<list>", "<ul>", "<items>"],
                        "correct_answer": "<ul>"
                    }
                ]
            elif step_data["language"] == "css":
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "How do you center an element horizontally in CSS?",
                        "options": ["align: center", "text-align: center", "margin: 0 auto", "position: center"],
                        "correct_answer": "margin: 0 auto"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "Which property sets the space between elements?",
                        "options": ["space", "margin", "padding", "gap"],
                        "correct_answer": "margin"
                    }
                ]
            else:  # javascript
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "How do you prevent a form from submitting in JavaScript?",
                        "options": ["stopSubmit()", "event.preventDefault()", "form.preventDefault()", "return false"],
                        "correct_answer": "event.preventDefault()"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "How do you add an element to the DOM in JavaScript?",
                        "options": ["document.add()", "element.append()", "document.addElement()", "document.createElement()"],
                        "correct_answer": "document.createElement()"
                    }
                ]
        else:  # python+streamlit
            if "basic" in step_data["title"].lower() or "setup" in step_data["title"].lower():
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "How do you display text in a Streamlit app?",
                        "options": ["print()", "st.write()", "st.text()", "st.display()"],
                        "correct_answer": "st.write()"
                    },
                    {
                        "question_id": "q2", 
                        "question_text": "What command creates a line chart in Streamlit?",
                        "options": ["st.line_chart()", "st.chart()", "st.plot()", "st.lineplot()"],
                        "correct_answer": "st.line_chart()"
                    }
                ]
            elif "control" in step_data["title"].lower() or "sidebar" in step_data["title"].lower():
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "How do you add a slider to a Streamlit app?",
                        "options": ["st.create_slider()", "st.slider()", "st.input_slider()", "st.range()"],
                        "correct_answer": "st.slider()"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "How do you add elements to the sidebar in Streamlit?",
                        "options": ["sidebar.element()", "st.sidebar", "st.sidebar.element()", "st.element(sidebar=True)"],
                        "correct_answer": "st.sidebar.element()"
                    }
                ]
            else:
                quiz_questions = [
                    {
                        "question_id": "q1",
                        "question_text": "Which library is commonly used for creating charts in Python?",
                        "options": ["chart", "pyplot", "matplotlib", "graphlib"],
                        "correct_answer": "matplotlib"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "What Pandas function loads data from a CSV file?",
                        "options": ["pd.load_csv()", "pd.read_csv()", "pd.open_csv()", "pd.csv()"],
                        "correct_answer": "pd.read_csv()"
                    }
                ]
            
        # Add quiz questions to step data
        step_data["quiz_questions"] = quiz_questions
        
        # Update session data with current step
        session["current_step"] = next_step
        
        return {
            "session_id": session_id,
            "response": json.dumps(step_data)
        }
    
    except ValidationError as ve:
        error_detail = f"Validation error: {str(ve)}"
        logger.error(error_detail)
        raise HTTPException(status_code=422, detail=error_detail)
    except Exception as e:
        error_detail = f"Error in test endpoint: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/terminate_streamlit")
async def terminate_streamlit(request: dict):
    """API endpoint to terminate a running Streamlit process"""
    execution_id = request.get("execution_id")
    if not execution_id:
        raise HTTPException(status_code=400, detail="Missing execution_id")
    
    success = terminate_streamlit_process(execution_id)
    return {"success": success}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

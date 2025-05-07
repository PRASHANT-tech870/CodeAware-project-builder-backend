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

class QuestionRequest(BaseModel):
    question: str
    code: str
    project_type: str
    session_id: str

# Store user sessions
user_sessions = {}

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
    
    Please provide:
    1. A brief introduction to the project
    2. A clear breakdown of steps to complete it
    3. The first step with detailed explanation and starter code
    
    Format your response as a JSON object with these fields:
    - project_title: A catchy name for the project
    - project_description: A brief overview of what we're building
    - total_steps: Total number of steps to complete the project
    - steps: An array of step objects, where each step has:
      - title: Step title
      - description: Detailed explanation
      - code: Starter code for this step
      - expected_outcome: What should happen after completing this step
    
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
            "gemini_response": clean_response_text
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
        "gemini_response": response_text
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
        
        logger.debug(f"Processing next step {next_step} for session {session_id}")
        
        # Prepare prompt for Gemini based on the current step and user's code
        prompt = f"""
        I'm building a project using {step_request.project_type} with expertise level {step_request.expertise_level}.
        Project idea: {step_request.project_idea}
        
        I have completed step {step_request.current_step} and here is my code:
        ```
        {step_request.user_code or "No code provided"}
        ```
        
        {f"I have a question: {step_request.user_question}" if step_request.user_question else ""}
        
        Please provide step {next_step} with:
        1. Feedback on my current code (if provided)
        2. Detailed explanation of the next step
        3. Starter code for the next step
        
        Format your response as a JSON object with these fields:
        - step_number: The step number
        - title: Step title
        - feedback: Feedback on the user's current code
        - description: Detailed explanation of this step
        - code: Starter code for this step
        - expected_outcome: What should happen after completing this step
        
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

# Add a test endpoint that doesn't rely on Gemini API
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working correctly"}

@app.get("/")
async def root():
    return {"message": "AI-Guided Project Builder API is running"}

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
        
        # Create a static response based on project type
        if step_request.project_type == "html+css+js":
            step_data = {
                "step_number": next_step,
                "title": f"Step {next_step}: Adding CSS styles",
                "feedback": "Your HTML structure looks good! Now let's add some styles.",
                "description": "In this step, we'll create a CSS file to style our portfolio website.",
                "code": "/* General styles */\nbody {\n  font-family: Arial, sans-serif;\n  line-height: 1.6;\n  margin: 0;\n  padding: 0;\n  color: #333;\n}\n\nheader {\n  background-color: #4a89dc;\n  color: white;\n  padding: 1rem;\n  text-align: center;\n}\n\nnav ul {\n  display: flex;\n  justify-content: center;\n  list-style: none;\n  padding: 0;\n}\n\nnav ul li {\n  margin: 0 15px;\n}\n\nnav a {\n  color: white;\n  text-decoration: none;\n}\n\nmain {\n  max-width: 1200px;\n  margin: 0 auto;\n  padding: 20px;\n}\n\nsection {\n  margin-bottom: 40px;\n}\n\nfooter {\n  background-color: #333;\n  color: white;\n  text-align: center;\n  padding: 1rem;\n  margin-top: 2rem;\n}",
                "expected_outcome": "A styled portfolio website with improved visual appearance."
            }
        else:  # python+streamlit
            step_data = {
                "step_number": next_step,
                "title": f"Step {next_step}: Adding Interactive Elements",
                "feedback": "Your basic Streamlit app looks good! Now let's add some interactivity.",
                "description": "In this step, we'll add interactive widgets to our Streamlit dashboard.",
                "code": "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Set up the app title and description\nst.title('Interactive Data Dashboard')\nst.write('Welcome to my interactive data visualization dashboard!')\n\n# Add a sidebar with controls\nst.sidebar.header('Dashboard Controls')\n\n# Add a slider for selecting the number of data points\nnum_points = st.sidebar.slider('Number of data points', 5, 30, 10)\n\n# Add a selectbox for choosing a chart type\nchart_type = st.sidebar.selectbox(\n    'Select chart type',\n    ['Line Chart', 'Bar Chart', 'Scatter Plot']\n)\n\n# Create some demo data based on the slider value\ndata = pd.DataFrame({\n    'Date': pd.date_range(start='2023-01-01', periods=num_points),\n    'Values': np.random.randn(num_points).cumsum()\n})\n\n# Display the data\nst.subheader('Sample Data')\nst.dataframe(data)\n\n# Show different charts based on selection\nst.subheader(f'{chart_type}')\nif chart_type == 'Line Chart':\n    st.line_chart(data.set_index('Date'))\nelif chart_type == 'Bar Chart':\n    st.bar_chart(data.set_index('Date'))\nelse:  # Scatter Plot\n    fig, ax = plt.subplots()\n    ax.scatter(range(len(data)), data['Values'])\n    ax.set_xlabel('Index')\n    ax.set_ylabel('Values')\n    st.pyplot(fig)",
                "expected_outcome": "An interactive Streamlit dashboard with controls in the sidebar that allow users to adjust the visualization."
            }
        
        # Update session data
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

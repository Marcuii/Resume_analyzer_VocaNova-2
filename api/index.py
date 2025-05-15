from flask import Flask, request, jsonify
import pdfplumber
from openai import OpenAI
import logging
import warnings
import os
import re
from flask_cors import CORS

warnings.filterwarnings('ignore')
logging.getLogger("pdfminer").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, origins=["https://vocanova.vercel.app"])

# Use environment variable for safety
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-3.5-turbo"

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def parse_feedback_response(response_text):
    # Initialize the structure with empty values
    feedback = {
        "Overall Rating": "",
        "Summary": "",
        "Strengths": [],
        "Weaknesses": [],
        "ATS Compatibility Analysis": "",
        "Formatting and Readability": "",
        "Content and Impact": "",
        "Grammar and Clarity": ""
    }
    
    # Normalize line endings and split into lines
    lines = response_text.replace('\r\n', '\n').split('\n')
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith("**Overall Rating:"):
            feedback["Overall Rating"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif line.startswith("**Summary:"):
            feedback["Summary"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif line == "**Strengths:**":
            current_section = "Strengths"
        elif line == "**Weaknesses:**":
            current_section = "Weaknesses"
        elif line.startswith("**ATS Compatibility Analysis:"):
            feedback["ATS Compatibility Analysis"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif line.startswith("**Formatting and Readability:"):
            feedback["Formatting and Readability"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif line.startswith("**Content and Impact:"):
            feedback["Content and Impact"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif line.startswith("**Grammar and Clarity:"):
            feedback["Grammar and Clarity"] = line.split(":", 1)[1].strip().strip('*').strip()
            current_section = None
        elif current_section in ["Strengths", "Weaknesses"]:
            # Handle bullet points (both numbered and unnumbered)
            if re.match(r'^\d+\.', line) or line.startswith('-'):
                # Remove bullet/number and add to list
                item = re.sub(r'^\d+\.\s*|^-\s*', '', line).strip()
                if item:  # Only add if there's actual content
                    feedback[current_section].append(item)
    
    return feedback

def get_resume_feedback(resume_text):
    prompt = f"""You are a professional career advisor. Analyze the following resume and provide feedback in this EXACT format:

**Overall Rating:** [score]/100

**Summary:** [one paragraph summary in just 2 lines]

**Strengths:**
1. [strength 1]
2. [strength 2]
3. [strength 3]

**Weaknesses:**
1. [weakness 1]
2. [weakness 2]
3. [weakness 3]

**ATS Compatibility Analysis:** [analysis and give rate score,just 2 lines]

**Formatting and Readability:** [feedback in just 2 lines]

**Content and Impact:** [feedback in just 2 lines]

**Grammar and Clarity:** [feedback in just 2 lines]

IMPORTANT: Maintain this exact formatting with the double asterisks for section headers and numbered lists for strengths/weaknesses.

Resume:
{resume_text}"""
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful and professional career advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=1.0,
        max_tokens=1000,
        model=MODEL_NAME
    )
    
    raw_feedback = response.choices[0].message.content
    return parse_feedback_response(raw_feedback)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Resume Analyzer API Running"})

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)
        feedback = get_resume_feedback(resume_text)

        # Clean up the temporary file
        os.remove(file_path)

        return jsonify(feedback)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

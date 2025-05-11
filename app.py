from flask import Flask, request, jsonify
import pdfplumber
from openai import OpenAI
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.getLogger("pdfminer").setLevel(logging.ERROR)

app = Flask(__name__)

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

def get_resume_feedback(resume_text):
    prompt = f"""
You are a professional career advisor. Analyze the following resume and provide the following structured feedback:

1. **Overall Rating (out of 100)** — Evaluate the overall quality of the resume.
2. **Summary** — A short paragraph in 1 line summarizing the impression of the resume.
3. **Strengths** — What is working well in this resume? (3-5 bullet points)
4. **Weaknesses** — What is not working well in this resume? (3-5 bullet points)
5. **ATS compatibility analysis** — give an ATS match score for this resume, issue and fix? (each one in one line)
6. **Formating and readability** — what is the issue of formating and fix? (each one in one line)
7. **Content and impact** — what is the issue of content and fix? (each one in one line)
8. **Grammer and clarity** - what is issue of grammer and fix? (each one in one line)
Resume:
{resume_text}
"""
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
    return response.choices[0].message.content

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

        return jsonify({'feedback': feedback})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel entry point
def handler(environ, start_response):
    return app(environ, start_response)

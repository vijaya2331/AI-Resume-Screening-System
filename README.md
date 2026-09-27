# AI Resume Screening & Job Matching System

An AI-based web application that analyzes a resume against a job description and identifies matching and missing skills.

## Features

- Upload resume in PDF format
- Extract text from the uploaded resume
- Enter or paste a job description
- Calculate resume-job similarity score using TF-IDF and cosine similarity
- Identify matching skills
- Suggest missing skills from the job description
- Simple and user-friendly Streamlit interface

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- PyPDF
- TF-IDF
- Cosine Similarity

## How It Works

```text
Resume PDF
    ↓
Text Extraction
    ↓
Text Processing
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Resume Match Score
    ↓
Skill Matching
    ↓
Suggested Skills

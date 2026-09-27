import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 20px 0 30px 0;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #777;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 12px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    background-color: #fafafa;
    margin-top: 20px;
}

.score {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    margin: 10px 0;
}

.skill-box {
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 8px;
    background-color: white;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SKILLS DATABASE
# --------------------------------------------------

skill_list = [
    "python",
    "java",
    "javascript",
    "c++",
    "sql",
    "html",
    "css",
    "react",
    "node.js",
    "flask",
    "django",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "data science",
    "data analysis",
    "data preprocessing",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "web scraping",
    "git",
    "github",
    "mongodb",
    "mysql",
    "power bi",
    "excel"
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

<div class="hero-title">
📄 AI Resume Screening System
</div>

<div class="hero-subtitle">
Analyze your resume against a job description and identify skill gaps.
</div>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

left_col, right_col = st.columns(2)

# Resume Upload
with left_col:

    st.markdown(
        '<div class="section-title">📄 Upload Resume</div>',
        unsafe_allow_html=True
    )

    resume_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )

# Job Description
with right_col:

    st.markdown(
        '<div class="section-title">💼 Job Description</div>',
        unsafe_allow_html=True
    )

    job_description = st.text_area(
        "Paste the job description here",
        height=220,
        placeholder="Paste the complete job description..."
    )

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

st.write("")

analyze = st.button(
    "🔍 Analyze Resume",
    use_container_width=True
)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:

    if resume_file is None:

        st.warning("Please upload a resume PDF.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        # ------------------------------------------
        # EXTRACT RESUME TEXT
        # ------------------------------------------

        reader = PdfReader(resume_file)

        resume_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                resume_text += text + " "

        # ------------------------------------------
        # TF-IDF SIMILARITY
        # ------------------------------------------

        documents = [
            resume_text,
            job_description
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        match_score = similarity[0][0] * 100

        # ------------------------------------------
        # SKILL EXTRACTION
        # ------------------------------------------

        resume_lower = resume_text.lower()

        job_lower = job_description.lower()

        resume_skills = []

        job_skills = []

        for skill in skill_list:

            pattern = r"\b" + re.escape(skill) + r"\b"

            if re.search(pattern, resume_lower):

                resume_skills.append(skill)

            if re.search(pattern, job_lower):

                job_skills.append(skill)

        # ------------------------------------------
        # MATCHING SKILLS
        # ------------------------------------------

        matching_skills = [
            skill
            for skill in job_skills
            if skill in resume_skills
        ]

        # ------------------------------------------
        # SUGGESTED SKILLS
        # ------------------------------------------

        suggested_skills = [
            skill
            for skill in job_skills
            if skill not in resume_skills
        ]

        # ------------------------------------------
        # RESULTS
        # ------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "<h2 style='text-align:center;'>📊 Resume Analysis</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="score">
                {match_score:.2f}%
            </div>

            <p style="text-align:center;">
                Resume Match Score
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------
        # SKILLS RESULT
        # ------------------------------------------

        st.write("")

        skill_col1, skill_col2 = st.columns(2)

        # Matching Skills
        with skill_col1:

            st.subheader("✅ Matching Skills")

            if matching_skills:

                for skill in matching_skills:

                    st.markdown(
                        f"""
                        <div class="skill-box">
                        ✅ {skill.title()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No matching skills detected."
                )

        # Suggested Skills
        with skill_col2:

            st.subheader("📌 Suggested Skills")

            if suggested_skills:

                for skill in suggested_skills:

                    st.markdown(
                        f"""
                        <div class="skill-box">
                        📌 {skill.title()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.success(
                    "No additional skills detected."
                )

        # ------------------------------------------
        # RESUME TEXT
        # ------------------------------------------

        st.write("")

        with st.expander("📄 View Extracted Resume Text"):

            st.text_area(
                "Resume Content",
                resume_text,
                height=300
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
AI Resume Screening & Job Matching System
</div>
""", unsafe_allow_html=True)
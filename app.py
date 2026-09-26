import streamlit as st
import re
import sqlite3
import json
from datetime import datetime
import pandas as pd
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 25px;
}

.score-box {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.score-number {
    font-size: 45px;
    font-weight: 700;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
}

.login-box {
    max-width: 450px;
    margin: 40px auto;
    padding: 30px;
    border: 1px solid #ddd;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOGIN SETTINGS
# =========================================================

LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "1234"


# =========================================================
# DATABASE
# =========================================================

DB_NAME = "resume_history.db"


def init_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            resume_name TEXT NOT NULL,
            final_score REAL NOT NULL,
            skill_score REAL NOT NULL,
            nlp_score REAL NOT NULL,
            matching_skills TEXT,
            missing_skills TEXT,
            resume_skills TEXT,
            strengths TEXT,
            improvements TEXT,
            suggestions TEXT,
            job_description TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(
    resume_name,
    final_score,
    skill_score,
    nlp_score,
    matching_skills,
    missing_skills,
    resume_skills,
    strengths,
    improvements,
    suggestions,
    job_description
):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis_history (
            username,
            analysis_date,
            resume_name,
            final_score,
            skill_score,
            nlp_score,
            matching_skills,
            missing_skills,
            resume_skills,
            strengths,
            improvements,
            suggestions,
            job_description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        st.session_state.username,
        datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        resume_name,
        final_score,
        skill_score,
        nlp_score,
        json.dumps(matching_skills),
        json.dumps(missing_skills),
        json.dumps(resume_skills),
        json.dumps(strengths),
        json.dumps(improvements),
        json.dumps(suggestions),
        job_description
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("""
        SELECT
            id,
            analysis_date,
            resume_name,
            final_score,
            skill_score,
            nlp_score
        FROM analysis_history
        WHERE username = ?
        ORDER BY id DESC
    """, conn, params=(st.session_state.username,))

    conn.close()

    return df


def get_analysis_by_id(analysis_id):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM analysis_history
        WHERE id = ? AND username = ?
    """, (analysis_id, st.session_state.username))

    row = cursor.fetchone()

    conn.close()

    return row


init_database()


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

if "page" not in st.session_state:
    st.session_state.page = "Analyzer"


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="main-title">📄 AI Resume Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Login to analyze resumes and manage your analysis history'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown("### 🔐 Login")

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        login = st.button(
            "🔐 Login",
            use_container_width=True
        )

        if login:

            if (
                username == LOGIN_USERNAME
                and password == LOGIN_PASSWORD
            ):

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "Analyzer"

                st.success("✅ Login successful!")

                st.rerun()

            else:

                st.error(
                    "❌ Invalid username or password."
                )

        st.info(
            "Demo Login\n\n"
            "Username: admin\n"
            "Password: 1234"
        )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📄 AI Resume Analyzer")

    st.write(
        f"👤 Logged in as: **{st.session_state.username}**"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Analyzer",
            "Analysis History"
        ],
        index=(
            0
            if st.session_state.page == "Analyzer"
            else 1
        )
    )

    st.session_state.page = page

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_analysis = None
        st.session_state.page = "Analyzer"

        st.rerun()


# =========================================================
# HISTORY PAGE
# =========================================================

if st.session_state.page == "Analysis History":

    st.markdown(
        '<div class="main-title">📋 Analysis History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'View your previous resume analyses'
        '</div>',
        unsafe_allow_html=True
    )

    history_df = get_history()

    if history_df.empty:

        st.info(
            "📭 No analysis history found yet. "
            "Analyze a resume first."
        )

    else:

        display_df = history_df.copy()

        display_df["final_score"] = (
            display_df["final_score"].map(
                lambda x: f"{x:.2f}%"
            )
        )

        display_df["skill_score"] = (
            display_df["skill_score"].map(
                lambda x: f"{x:.2f}%"
            )
        )

        display_df["nlp_score"] = (
            display_df["nlp_score"].map(
                lambda x: f"{x:.2f}%"
            )
        )

        display_df = display_df.rename(columns={
            "id": "ID",
            "analysis_date": "Date",
            "resume_name": "Resume",
            "final_score": "Final Score",
            "skill_score": "Skill Score",
            "nlp_score": "NLP Score"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("🔎 View Previous Analysis")

        selected_id = st.selectbox(
            "Select analysis",
            history_df["id"].tolist(),
            format_func=lambda x: (
                f"Analysis #{x} - "
                f"{history_df.loc[history_df['id'] == x, 'resume_name'].iloc[0]} - "
                f"{history_df.loc[history_df['id'] == x, 'final_score'].iloc[0]:.2f}%"
            )
        )

        row = get_analysis_by_id(selected_id)

        if row:

            (
                analysis_id,
                username,
                analysis_date,
                resume_name,
                final_score,
                skill_score,
                nlp_score,
                matching_json,
                missing_json,
                resume_skills_json,
                strengths_json,
                improvements_json,
                suggestions_json,
                saved_job_description
            ) = row

            matching_skills = json.loads(matching_json)
            missing_skills = json.loads(missing_json)
            resume_skills = json.loads(resume_skills_json)
            strengths = json.loads(strengths_json)
            improvements = json.loads(improvements_json)
            suggestions = json.loads(suggestions_json)

            st.success(
                f"📄 **{resume_name}** | "
                f"Analyzed: **{analysis_date}**"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🎯 Final Match",
                    f"{final_score:.2f}%"
                )

            with col2:
                st.metric(
                    "🛠️ Skill Match",
                    f"{skill_score:.2f}%"
                )

            with col3:
                st.metric(
                    "🧠 NLP Similarity",
                    f"{nlp_score:.2f}%"
                )

            st.subheader("✅ Matching Skills")

            if matching_skills:
                st.write(", ".join(matching_skills))
            else:
                st.info("No matching skills.")

            st.subheader("❌ Missing Skills")

            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.success("No missing skills.")

            st.subheader("📄 Skills Found in Resume")

            if resume_skills:
                st.write(", ".join(resume_skills))
            else:
                st.info("No predefined skills detected.")

            st.subheader("💪 Resume Strengths")

            for item in strengths:
                st.success("✓ " + item)

            st.subheader("⚠️ Areas to Improve")

            for item in improvements:
                st.warning("• " + item)

            st.subheader("💡 Suggestions")

            for item in suggestions:
                st.info("💡 " + item)

            st.subheader("💼 Saved Job Description")

            st.text_area(
                "Job Description",
                saved_job_description,
                height=180,
                disabled=True
            )

    st.divider()

    st.markdown(
        '<div class="footer">'
        'AI Resume Analyzer | Analysis History'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# ANALYZER PAGE
# =========================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume, compare job requirements, and get improvement suggestions'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "Python",
    "Java",
    "C++",
    "C",
    "JavaScript",
    "HTML",
    "CSS",
    "React",
    "Django",
    "Flask",
    "FastAPI",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Git",
    "GitHub",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "AI",
    "Data Science",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "OpenCV",
    "NLP",
    "Computer Vision",
    "Excel",
    "Power BI",
    "Tableau",
    "AWS",
    "Azure",
    "Docker",
    "Linux",
    "REST API",
    "API",
    "Communication",
    "Problem Solving"
]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# SKILL DETECTION
# =========================================================

def find_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text_lower):

            found_skills.append(skill)

    return found_skills


# =========================================================
# SKILL MATCH SCORE
# =========================================================

def calculate_match(resume_skills, job_skills):

    if not job_skills:
        return 0

    matching = set(resume_skills) & set(job_skills)

    score = (
        len(matching) /
        len(job_skills)
    ) * 100

    return round(score, 2)


# =========================================================
# NLP SIMILARITY
# =========================================================

def calculate_nlp_similarity(resume_text, job_description):

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    if not resume_text or not job_description:
        return 0

    try:

        documents = [
            resume_text.lower(),
            job_description.lower()
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        score = similarity * 100

        return round(score, 2)

    except Exception:

        return 0


# =========================================================
# FINAL MATCH SCORE
# =========================================================

def calculate_final_score(skill_score, nlp_score):

    final_score = (
        (skill_score * 0.60) +
        (nlp_score * 0.40)
    )

    return round(final_score, 2)


# =========================================================
# RESUME STRENGTH ANALYSIS
# =========================================================

def analyze_resume_strength(
    resume_text,
    resume_skills,
    job_skills,
    matching_skills,
    missing_skills
):

    strengths = []
    improvements = []
    suggestions = []

    text_lower = resume_text.lower()

    if len(resume_skills) >= 8:

        strengths.append(
            "Your resume contains a good number of technical skills."
        )

    elif len(resume_skills) >= 4:

        strengths.append(
            "Your resume contains several technical skills."
        )

    else:

        improvements.append(
            "Your resume contains relatively few detected technical skills."
        )

        suggestions.append(
            "Add relevant technical skills that you genuinely have experience with."
        )

    if matching_skills:

        strengths.append(
            "Your resume contains skills that match the target job."
        )

    if missing_skills:

        improvements.append(
            "Some skills mentioned in the job description were not detected in your resume."
        )

        suggestions.append(
            "If you have experience with the missing skills, mention them clearly in your resume."
        )

    if "project" in text_lower or "projects" in text_lower:

        strengths.append(
            "Your resume includes project-related information."
        )

    else:

        improvements.append(
            "A dedicated projects section was not detected."
        )

        suggestions.append(
            "Add 2–3 relevant academic or personal projects with technologies used."
        )

    education_words = [
        "education",
        "bachelor",
        "b.tech",
        "btech",
        "degree",
        "university",
        "college"
    ]

    if any(word in text_lower for word in education_words):

        strengths.append(
            "Education information was detected in the resume."
        )

    else:

        improvements.append(
            "Education information was not clearly detected."
        )

        suggestions.append(
            "Include your degree, college/university, and graduation year."
        )

    experience_words = [
        "experience",
        "internship",
        "intern",
        "work experience"
    ]

    if any(word in text_lower for word in experience_words):

        strengths.append(
            "Experience or internship information was detected."
        )

    else:

        improvements.append(
            "Work experience or internship information was not clearly detected."
        )

        suggestions.append(
            "If applicable, add internships, training, freelance work, or practical experience."
        )

    if "@" in resume_text:

        strengths.append(
            "An email address was detected."
        )

    else:

        improvements.append(
            "An email address was not detected."
        )

        suggestions.append(
            "Make sure your professional email address is included."
        )

    if not strengths:

        strengths.append(
            "Resume text was successfully extracted and analyzed."
        )

    if not improvements:

        improvements.append(
            "No major improvement areas were detected by this analyzer."
        )

    if not suggestions:

        suggestions.append(
            "Continue tailoring your resume to each job description."
        )

    return strengths, improvements, suggestions


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf(
    final_score,
    skill_score,
    nlp_score,
    matching_skills,
    missing_skills,
    resume_skills,
    strengths,
    improvements,
    suggestions
):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    y = height - 50

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        y,
        "AI Resume Analyzer Report"
    )

    y -= 40

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        f"Final Match Score: {final_score}%"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        50,
        y,
        f"Skill Match Score: {skill_score}%"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"NLP Similarity Score: {nlp_score}%"
    )

    y -= 35

    def add_section(title, items):

        nonlocal y

        if y < 100:

            pdf.showPage()

            y = height - 50

        pdf.setFont(
            "Helvetica-Bold",
            13
        )

        pdf.drawString(
            50,
            y,
            title
        )

        y -= 25

        pdf.setFont(
            "Helvetica",
            10
        )

        for item in items:

            item = str(item)

            if len(item) > 90:
                item = item[:87] + "..."

            pdf.drawString(
                65,
                y,
                "- " + item
            )

            y -= 18

            if y < 60:

                pdf.showPage()

                y = height - 50

                pdf.setFont(
                    "Helvetica",
                    10
                )

        y -= 10

    add_section(
        "Matching Skills",
        matching_skills if matching_skills
        else ["No matching skills found."]
    )

    add_section(
        "Missing Skills",
        missing_skills if missing_skills
        else ["No missing skills found."]
    )

    add_section(
        "Skills Found in Resume",
        resume_skills if resume_skills
        else ["No predefined skills detected."]
    )

    add_section(
        "Resume Strengths",
        strengths
    )

    add_section(
        "Areas to Improve",
        improvements
    )

    add_section(
        "Suggestions",
        suggestions
    )

    pdf.save()

    buffer.seek(0)

    return buffer


# =========================================================
# RESET ANALYZER
# =========================================================

if st.button(
    "🔄 Reset Analyzer",
    use_container_width=False
):

    st.session_state.current_analysis = None

    st.rerun()


# =========================================================
# UPLOAD RESUME
# =========================================================

st.markdown(
    '<div class="section-title">1️⃣ Upload Resume</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:

    try:

        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        if resume_text.strip():

            st.success(
                "✅ Resume uploaded successfully!"
            )

            with st.expander(
                "📄 View Extracted Resume Text"
            ):

                st.text_area(
                    "Resume Text",
                    clean_text(resume_text),
                    height=250
                )

        else:

            st.error(
                "The PDF does not contain readable text."
            )

    except Exception as error:

        st.error(
            f"Could not read the PDF: {error}"
        )


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.markdown(
    '<div class="section-title">2️⃣ Enter Job Description</div>',
    unsafe_allow_html=True
)

job_description = st.text_area(
    "Paste the job description below",
    height=220,
    placeholder="Example: We are looking for a Python Developer..."
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Resume",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not uploaded_file:

        st.error(
            "⚠️ Please upload your resume first."
        )

    elif not resume_text.strip():

        st.error(
            "⚠️ Could not extract readable text from the resume."
        )

    elif not job_description.strip():

        st.error(
            "⚠️ Please enter a job description."
        )

    else:

        resume_skills = find_skills(
            resume_text
        )

        job_skills = find_skills(
            job_description
        )

        matching_skills = sorted(
            list(
                set(resume_skills)
                &
                set(job_skills)
            )
        )

        missing_skills = sorted(
            list(
                set(job_skills)
                -
                set(resume_skills)
            )
        )

        skill_score = calculate_match(
            resume_skills,
            job_skills
        )

        nlp_score = calculate_nlp_similarity(
            resume_text,
            job_description
        )

        final_score = calculate_final_score(
            skill_score,
            nlp_score
        )

        strengths, improvements, suggestions = (
            analyze_resume_strength(
                resume_text,
                resume_skills,
                job_skills,
                matching_skills,
                missing_skills
            )
        )

        # Save the complete current analysis in session state.
        st.session_state.current_analysis = {
            "resume_name": uploaded_file.name,
            "final_score": final_score,
            "skill_score": skill_score,
            "nlp_score": nlp_score,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "resume_skills": resume_skills,
            "strengths": strengths,
            "improvements": improvements,
            "suggestions": suggestions,
            "job_description": job_description
        }

        # Save to SQLite history.
        save_analysis(
            uploaded_file.name,
            final_score,
            skill_score,
            nlp_score,
            matching_skills,
            missing_skills,
            resume_skills,
            strengths,
            improvements,
            suggestions,
            job_description
        )

        st.success(
            "✅ Analysis completed and saved to Analysis History."
        )


# =========================================================
# DISPLAY CURRENT ANALYSIS
# =========================================================

analysis = st.session_state.current_analysis

if analysis:

    final_score = analysis["final_score"]
    skill_score = analysis["skill_score"]
    nlp_score = analysis["nlp_score"]
    matching_skills = analysis["matching_skills"]
    missing_skills = analysis["missing_skills"]
    resume_skills = analysis["resume_skills"]
    strengths = analysis["strengths"]
    improvements = analysis["improvements"]
    suggestions = analysis["suggestions"]

    st.markdown(
        '<div class="section-title">3️⃣ Resume Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="score-box">
            <div>🎯 Final Job Match Score</div>
            <div class="score-number">
                {final_score}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(
        min(100, int(final_score))
    )

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        st.metric(
            "🧠 NLP Similarity",
            f"{nlp_score}%"
        )

    with score_col2:

        st.metric(
            "🛠️ Skill Match",
            f"{skill_score}%"
        )

    st.caption(
        "Final score combines predefined skill matching "
        "and TF-IDF based text similarity."
    )

    # =====================================================
    # VISUAL DASHBOARD
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Resume Dashboard"
    )

    dashboard_col1, dashboard_col2 = st.columns(2)

    with dashboard_col1:

        st.markdown(
            "### 🎯 Score Breakdown"
        )

        score_data = pd.DataFrame({
            "Score Type": [
                "Final Match",
                "NLP Similarity",
                "Skill Match"
            ],
            "Score": [
                final_score,
                nlp_score,
                skill_score
            ]
        })

        st.bar_chart(
            score_data.set_index("Score Type")
        )

    with dashboard_col2:

        st.markdown(
            "### 🛠️ Skill Comparison"
        )

        skill_data = pd.DataFrame({
            "Category": [
                "Matching Skills",
                "Missing Skills"
            ],
            "Count": [
                len(matching_skills),
                len(missing_skills)
            ]
        })

        st.bar_chart(
            skill_data.set_index("Category")
        )

    # =====================================================
    # SKILL METRICS
    # =====================================================

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Resume Skills",
            len(resume_skills)
        )

    with col2:

        st.metric(
            "💼 Job Skills",
            len(find_skills(analysis["job_description"]))
        )

    with col3:

        st.metric(
            "✅ Matching",
            len(matching_skills)
        )

    with col4:

        st.metric(
            "❌ Missing",
            len(missing_skills)
        )

    st.divider()

    # =====================================================
    # MATCHING / MISSING
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "✅ Matching Skills"
        )

        if matching_skills:

            for skill in matching_skills:

                st.success(
                    "✓ " + skill
                )

        else:

            st.info(
                "No matching skills found."
            )

    with col2:

        st.subheader(
            "❌ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:

                st.warning(
                    "• " + skill
                )

        else:

            st.success(
                "No missing skills!"
            )

    st.divider()

    # =====================================================
    # RESUME STRENGTHS
    # =====================================================

    st.subheader(
        "💪 Resume Strengths"
    )

    for strength in strengths:

        st.success(
            "✓ " + strength
        )

    # =====================================================
    # AREAS TO IMPROVE
    # =====================================================

    st.subheader(
        "⚠️ Areas to Improve"
    )

    for improvement in improvements:

        st.warning(
            "• " + improvement
        )

    # =====================================================
    # SUGGESTIONS
    # =====================================================

    st.subheader(
        "💡 Personalized Suggestions"
    )

    for suggestion in suggestions:

        st.info(
            "💡 " + suggestion
        )

    st.divider()

    # =====================================================
    # RESUME SKILLS
    # =====================================================

    st.subheader(
        "📄 Skills Found in Resume"
    )

    if resume_skills:

        st.write(
            ", ".join(resume_skills)
        )

    else:

        st.info(
            "No predefined skills detected."
        )

    # =====================================================
    # JOB SKILLS
    # =====================================================

    job_skills = find_skills(
        analysis["job_description"]
    )

    st.subheader(
        "💼 Skills Required by Job"
    )

    if job_skills:

        st.write(
            ", ".join(job_skills)
        )

    else:

        st.info(
            "No predefined skills detected."
        )

    # =====================================================
    # NLP ANALYSIS
    # =====================================================

    st.subheader(
        "🧠 NLP Analysis"
    )

    st.info(
        f"""
The NLP model compared the language used in your resume
with the language used in the job description.

NLP Similarity Score: {nlp_score}%

This score is calculated using TF-IDF and cosine similarity.
It measures textual similarity between the resume and job description.
"""
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader(
        "📝 Analysis Summary"
    )

    if final_score >= 80:

        st.success(
            f"Your resume shows a strong overall match "
            f"with the job description. Final score: "
            f"{final_score}%."
        )

    elif final_score >= 50:

        st.info(
            f"Your resume shows a moderate overall match. "
            f"Consider improving the missing skills and "
            f"tailoring your resume. Final score: "
            f"{final_score}%."
        )

    else:

        st.warning(
            f"Your resume shows a lower overall match with "
            f"this job description. Consider tailoring your "
            f"resume to relevant requirements. Final score: "
            f"{final_score}%."
        )

    # =====================================================
    # PDF DOWNLOAD
    # =====================================================

    st.divider()

    st.subheader(
        "📥 Download Complete Report"
    )

    pdf_file = create_pdf(
        final_score,
        skill_score,
        nlp_score,
        matching_skills,
        missing_skills,
        resume_skills,
        strengths,
        improvements,
        suggestions
    )

    st.download_button(
        "📄 Download PDF Report",
        data=pdf_file,
        file_name="resume_analysis_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'AI Resume Analyzer | Python + Streamlit + NLP + SQLite'
    '</div>',
    unsafe_allow_html=True
)

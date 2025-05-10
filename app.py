import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import spacy
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import tempfile
import os

# Initialize page config

st.set_page_config(
    page_title="Resume & Job Description Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download required resources

@st.cache_resource
def download_resources():
    # Download NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab',quiet=True)
    

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # If model not found, download it
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return nlp

nlp = download_resources()

class ResumeParser:
    """Class to parse resume from different file formats"""

    @staticmethod
    def parse_resume(uploaded_file):
        """Parse resume based on file extension"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            return ResumeParser.parse_pdf(uploaded_file)
        elif file_extension == '.docx':
            return ResumeParser.parse_docx(uploaded_file)
        elif file_extension == '.txt':
            return ResumeParser.parse_txt(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    @staticmethod
    def parse_pdf(uploaded_file):
        """Extract text from PDF file"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        os.unlink(temp_file_path)
        return text

    @staticmethod
    def parse_docx(uploaded_file):
        """Extract text from DOCX file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        text = docx2txt.process(temp_file_path)
        os.unlink(temp_file_path)
        return text

    @staticmethod
    def parse_txt(uploaded_file):
        """Extract text from TXT file"""
        return uploaded_file.getvalue().decode('utf-8')

class AdvancedNLP:
    """Class for advanced NLP features"""

    def __init__(self):
        # Load custom entities and phrases
        self.education_keywords = [
            "education", "academic background", "degree", "university", "college", "school",
            "bachelor", "master", "phd", "doctorate", "diploma", "certification"
        ]
        
        self.experience_keywords = [
            "experience", "work history", "employment", "career", "job", "position",
            "professional background", "work experience"
        ]
        
        self.skills_keywords = [
            "skills", "technical skills", "competencies", "expertise", "proficiency",
            "qualifications", "abilities", "core competencies"
        ]
        
        # Load pre-defined skills classification
        self.skills_categories = {
            "programming": [
                "python", "java", "javascript", "c++", "ruby", "php", "swift", "kotlin", "go",
                "scala", "rust", "typescript", "html", "css", "perl", "shell", "bash"
            ],
            "databases": [
                "sql", "mysql", "postgresql", "mongodb", "nosql", "oracle", "sqlite", "redis",
                "cassandra", "dynamodb", "mariadb", "firebase"
            ],
            "frameworks": [
                "django", "flask", "react", "angular", "vue", "spring", "express", "rails",
                "laravel", "asp.net", "tensorflow", "pytorch", "bootstrap", "jquery"
            ],
            "cloud": [
                "aws", "azure", "gcp", "google cloud", "cloud computing", "s3", "ec2", "lambda",
                "cloud storage", "kubernetes", "docker", "containerization"
            ],
            "soft_skills": [
                "communication", "leadership", "teamwork", "problem solving", "critical thinking",
                "time management", "creativity", "adaptability", "collaboration"
            ]
        }

    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        doc = nlp(text)
        entities = {}
        
        # Extract standard entities
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities

    def extract_education(self, text):
        """Extract education section from resume"""
        sections = self._split_to_sections(text)
        education_section = ""
        
        for section, content in sections.items():
            if any(keyword in section.lower() for keyword in self.education_keywords):
                education_section += content + "\n"
        
        # Extract education entities
        education_data = []
        doc = nlp(education_section)
        
        # Look for degree patterns
        degree_patterns = [
            r"(B\.?S\.?|Bachelor of Science|Bachelor's) (?:degree )?(?:in )?([\w\s]+)",
            r"(B\.?A\.?|Bachelor of Arts|Bachelor's) (?:degree )?(?:in )?([\w\s]+)",
            r"(M\.?S\.?|Master of Science|Master's) (?:degree )?(?:in )?([\w\s]+)",
            r"(M\.?B\.?A\.?|Master of Business Administration)",
            r"(Ph\.?D\.?|Doctor of Philosophy) (?:(?:degree|doctorate) )?(?:in )?([\w\s]+)",
            r"(Associate'?s? (?:degree)?) (?:in )?([\w\s]+)"
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, education_section, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 1:
                    education_data.append({
                        "degree": match.group(1),
                        "field": match.group(2) if len(match.groups()) > 1 else ""
                    })
                else:
                    education_data.append({
                        "degree": match.group(1),
                        "field": ""
                    })
        
        # Look for university/college names
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(edu_term in ent.text.lower() for edu_term in 
                                          ["university", "college", "institute", "school"]):
                education_data.append({
                    "institution": ent.text
                })
        
        return education_data

    def extract_experience(self, text):
        """Extract work experience section from resume"""
        sections = self._split_to_sections(text)
        experience_section = ""
        
        for section, content in sections.items():
            if any(keyword in section.lower() for keyword in self.experience_keywords):
                experience_section += content + "\n"
        
        # Parse experience section to extract job details
        experience_data = []
        
        # Look for job title patterns
        job_patterns = [
            r"((?:Senior|Junior|Lead|Chief|Principal)?\s*[\w\s]+?(?:Engineer|Developer|Designer|Manager|Director|Analyst|Consultant|Administrator|Specialist))\s+(?:at|@|with|for)?\s+([\w\s&\-.,]+)(?:\s+\((\w+\s+\d{4}\s*(?:-|–|to)\s*(?:\w+\s+\d{4}|Present|Current))\))?",
            r"([\w\s]+\s+(?:Engineer|Developer|Designer|Manager|Director|Analyst|Consultant|Administrator|Specialist))\s*\|\s*([\w\s&\-.,]+)\s*\|\s*(\w+\s+\d{4}\s*(?:-|–|to)\s*(?:\w+\s+\d{4}|Present|Current))"
        ]
        
        for pattern in job_patterns:
            matches = re.finditer(pattern, experience_section, re.IGNORECASE)
            for match in matches:
                job_info = {
                    "title": match.group(1).strip() if match.group(1) else "",
                    "company": match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else "",
                    "duration": match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else ""
                }
                experience_data.append(job_info)
        
        return experience_data

    def extract_skills(self, text):
        """Extract skills using simple pattern matching"""
        # For simplicity, we'll use a list of skills from the categories
        all_skills = [skill for category in self.skills_categories.values() for skill in category]
        extracted_skills = set()
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        for skill in all_skills:
            # Use word boundary to match whole words only
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                extracted_skills.add(skill)
        
        return list(extracted_skills)

    def categorize_skills(self, skills):
        """Categorize skills into different categories"""
        categorized = {category: [] for category in self.skills_categories}
        uncategorized = []
        
        for skill in skills:
            categorized_flag = False
            for category, category_skills in self.skills_categories.items():
                if any(category_skill in skill.lower() for category_skill in category_skills):
                    categorized[category].append(skill)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                uncategorized.append(skill)
        
        # Add uncategorized skills
        if uncategorized:
            categorized["other"] = uncategorized
        
        return categorized

    def _split_to_sections(self, text):
        """Split resume text into sections based on headings"""
        # Common section headers pattern
        section_pattern = r'\n(?:\s*)([\w\s&]+?)(?:\s*)(?:\n|:)'
        
        # Find all potential section headers
        matches = re.finditer(section_pattern, "\n" + text)
        sections = {}
        current_section = "header"
        start_pos = 0
        for match in matches:
            # Extract content of the previous section
            section_content = text[start_pos:match.start()].strip()
            if section_content:
                sections[current_section] = section_content
            
            # Update for the next section
            group_text = match.group(1)
            current_section = group_text.strip() if group_text else "Unnamed Section"
            start_pos = match.end()
        
        # Add the last section
        if start_pos < len(text):
            sections[current_section] = text[start_pos:].strip()
        
        return sections

class EnhancedMatcher:
    """Enhanced version of resume-job matcher with advanced NLP features"""

    def __init__(self):
        self.parser = ResumeParser()
        self.nlp_processor = AdvancedNLP()
        self.tfidf_vectorizer = TfidfVectorizer()

    def parse_resume(self, resume_file):
        """Parse resume file"""
        resume_text = self.parser.parse_resume(resume_file)
        return resume_text

    def parse_job_description(self, job_file):
        """Parse job description file"""
        if isinstance(job_file, str):
            return job_file
        else:
            return self.parser.parse_resume(job_file)

    def advanced_analysis(self, resume_text, job_text):
        """Perform advanced analysis on resume and job description"""
        # Extract skills from job description
        job_skills = self.nlp_processor.extract_skills(job_text)
        
        # Extract skills from resume
        resume_skills = self.nlp_processor.extract_skills(resume_text)
        
        # Categorize skills
        categorized_job_skills = self.nlp_processor.categorize_skills(job_skills)
        categorized_resume_skills = self.nlp_processor.categorize_skills(resume_skills)
        
        # Extract education and experience from resume
        education = self.nlp_processor.extract_education(resume_text)
        experience = self.nlp_processor.extract_experience(resume_text)
        
        # Calculate TF-IDF similarity
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        
        # Prepare results
        results = {
            "similarity_score": similarity,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "matching_skills": list(set(resume_skills).intersection(set(job_skills))),
            "missing_skills": list(set(job_skills) - set(resume_skills)),
            "categorized_resume_skills": categorized_resume_skills,
            "categorized_job_skills": categorized_job_skills,
            "education": education,
            "experience": experience
        }
        
        return results

    def generate_advanced_recommendations(self, analysis_results):
        """Generate advanced recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Recommendations based on similarity score
        similarity = analysis_results["similarity_score"]
        if similarity < 40:
            recommendations.append({
                "category": "overall",
                "recommendation": "Your resume has a low match with this job description. Consider significant revisions.",
                "importance": "high"
            })
        elif similarity < 60:
            recommendations.append({
                "category": "overall",
                "recommendation": "Your resume has a moderate match. Some targeted improvements could increase your chances.",
                "importance": "medium"
            })
        elif similarity < 80:
            recommendations.append({
                "category": "overall",
                "recommendation": "Your resume has a good match with this job description. A few small additions could make it even stronger.",
                "importance": "low"
            })
        else:
            recommendations.append({
                "category": "overall",
                "recommendation": "Your resume has an excellent match with this job description!",
                "importance": "info"
            })
        
        # Recommendations based on missing skills
        missing_skills = analysis_results["missing_skills"]
        if missing_skills:
            missing_by_category = {}
            for skill in missing_skills:
                for category, skills in analysis_results["categorized_job_skills"].items():
                    if skill in skills:
                        if category not in missing_by_category:
                            missing_by_category[category] = []
                        missing_by_category[category].append(skill)
            
            # Generate recommendations by category
            for category, skills in missing_by_category.items():
                if category != "other" and len(skills) > 0:
                    skill_list = ", ".join(skills[:3])
                    if len(skills) > 3:
                        skill_list += f", and {len(skills) - 3} more"
                    
                    category_name = category.replace("_", " ").title()
                    recommendations.append({
                        "category": category,
                        "recommendation": f"Add {category_name} skills to your resume: {skill_list}",
                        "importance": "high" if len(skills) > 3 else "medium"
                    })
        
        # Recommendations based on experience
        if not analysis_results["experience"]:
            recommendations.append({
                "category": "experience",
                "recommendation": "Your work experience section could not be detected or parsed. Make sure it's clearly labeled and formatted.",
                "importance": "high"
            })
        
        # Recommendations based on education
        if not analysis_results["education"]:
            recommendations.append({
                "category": "education",
                "recommendation": "Your education section could not be detected or parsed. Make sure it's clearly labeled and formatted.",
                "importance": "medium"
            })
        
        return recommendations

# Streamlit UI

st.title("📄 Resume & Job Description Matcher")
st.markdown("""
Upload your resume and job description to see how well your resume matches the job requirements.
Get detailed analysis and recommendations to improve your chances of landing the job!
""")

# Create tabs for different functionalities

tab1, tab2, tab3 = st.tabs(["File Upload & Analysis", "Results Dashboard", "Recommendations"])

# Initialize session state for storing results

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "job_text" not in st.session_state:
    st.session_state.job_text = None

# File Upload Tab

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader("Choose your resume file", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT")
        
        if resume_file is not None:
            try:
                st.success(f"File uploaded: {resume_file.name}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        st.subheader("Job Description")
        job_option = st.radio("Job Description Source", ["Upload File", "Paste Text"])
        
        if job_option == "Upload File":
            job_file = st.file_uploader("Choose job description file", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT")
        else:
            job_text_input = st.text_area("Paste job description here", height=300)
            job_file = job_text_input if job_text_input else None

    # Analysis Button
    if st.button("Analyze Match"):
        if resume_file is None:
            st.error("Please upload your resume.")
        elif job_file is None:
            st.error("Please provide a job description.")
        else:
            with st.spinner("Analyzing your resume and job description..."):
                try:
                    matcher = EnhancedMatcher()
                    
                    # Parse files
                    resume_text = matcher.parse_resume(resume_file)
                    st.session_state.resume_text = resume_text
                    
                    if job_option == "Upload File":
                        job_text = matcher.parse_job_description(job_file)
                    else:
                        job_text = job_file
                    
                    # FIX: Check if job_text is None and handle it
                    if job_text is None:
                        st.error("Could not parse job description. Please make sure it's not empty.")
                    else:
                        st.session_state.job_text = job_text
                        
                        # Perform analysis
                        results = matcher.advanced_analysis(resume_text, job_text)
                        st.session_state.analysis_results = results
                        
                        # Generate recommendations
                        recommendations = matcher.generate_advanced_recommendations(results)
                        st.session_state.recommendations = recommendations
                        
                        st.success("Analysis complete! Check the Results and Recommendations tabs.")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Results Dashboard Tab

with tab2:
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results

        # Display match score with gauge
        st.subheader("Match Score")
        match_score = results["similarity_score"]
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create a gauge chart for the match score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = match_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Resume-Job Match"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': match_score
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if match_score < 40:
                st.error("Low match. Significant improvements needed.")
            elif match_score < 60:
                st.warning("Moderate match. Some improvements recommended.")
            elif match_score < 80:
                st.info("Good match. Minor improvements could help.")
            else:
                st.success("Excellent match!")
        
        with col2:
            # Skills match summary
            st.subheader("Skills Summary")
            matching = len(results["matching_skills"])
            missing = len(results["missing_skills"])
            resume_extra = len(results["resume_skills"]) - matching
            
            # Create a pie chart for skills comparison
            fig = go.Figure(data=[go.Pie(
                labels=['Matching Skills', 'Missing Skills', 'Extra Skills in Resume'],
                values=[matching, missing, resume_extra],
                hole=.3,
                marker_colors=['green', 'red', 'gray']
            )])
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Skills Details
        st.subheader("Skills Details")
        skills_tab1, skills_tab2, skills_tab3 = st.tabs(["Matching Skills", "Missing Skills", "All Skills by Category"])
        
        with skills_tab1:
            if results["matching_skills"]:
                st.write("Skills present in both your resume and the job description:")
                matching_skills_df = pd.DataFrame({"Matching Skills": results["matching_skills"]})
                st.dataframe(matching_skills_df, use_container_width=True)
            else:
                st.warning("No matching skills found.")
        
        with skills_tab2:
            if results["missing_skills"]:
                st.write("Skills mentioned in the job description but not found in your resume:")
                missing_skills_df = pd.DataFrame({"Missing Skills": results["missing_skills"]})
                st.dataframe(missing_skills_df, use_container_width=True)
            else:
                st.success("No missing skills! Your resume covers all skills mentioned in the job description.")
        
        with skills_tab3:
            # Create a horizontal bar chart comparing resume skills vs job skills by category
            categories = list(results["categorized_job_skills"].keys())
            resume_counts = []
            job_counts = []
            
            for category in categories:
                resume_category_skills = results["categorized_resume_skills"].get(category, [])
                job_category_skills = results["categorized_job_skills"].get(category, [])
                
                resume_counts.append(len(resume_category_skills))
                job_counts.append(len(job_category_skills))
            
            # Create the figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=[cat.replace("_", " ").title() for cat in categories],
                x=resume_counts,
                name='Skills in Resume',
                orientation='h',
                marker=dict(color='royalblue')
            ))
            
            fig.add_trace(go.Bar(
                y=[cat.replace("_", " ").title() for cat in categories],
                x=job_counts,
                name='Skills in Job Description',
                orientation='h',
                marker=dict(color='darkorange')
            ))
            
            fig.update_layout(
                title='Skills by Category',
                barmode='group',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed skills by category
            for category in categories:
                if category in results["categorized_job_skills"] and len(results["categorized_job_skills"][category]) > 0:
                    with st.expander(f"{category.replace('_', ' ').title()} Skills"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**In Your Resume:**")
                            if category in results["categorized_resume_skills"] and results["categorized_resume_skills"][category]:
                                for skill in results["categorized_resume_skills"][category]:
                                    st.write(f"- {skill}")
                            else:
                                st.write("None found")
                        
                        with col2:
                            st.write("**In Job Description:**")
                            if results["categorized_job_skills"][category]:
                                for skill in results["categorized_job_skills"][category]:
                                    if skill in results["categorized_resume_skills"].get(category, []):
                                        st.write(f"- {skill} ✅")
                                    else:
                                        st.write(f"- {skill} ❌")
                            else:
                                st.write("None found")
        
        # Education & Experience
        exp_edu_col1, exp_edu_col2 = st.columns(2)
        
        with exp_edu_col1:
            st.subheader("Education")
            if results["education"]:
                for edu in results["education"]:
                    if "institution" in edu:
                        st.write(f"**Institution:** {edu['institution']}")
                    if "degree" in edu:
                        degree_text = f"**Degree:** {edu['degree']}"
                        if "field" in edu and edu["field"]:
                            degree_text += f" in {edu['field']}"
                        st.write(degree_text)
                    st.write("---")
            else:
                st.warning("No education information extracted from your resume.")
        
        with exp_edu_col2:
            st.subheader("Experience")
            if results["experience"]:
                for exp in results["experience"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "title" in exp:
                            st.write(f"**{exp['title']}**")
                        if "company" in exp:
                            st.write(f"*{exp['company']}*")
                    with col2:
                        if "duration" in exp:
                            st.write(f"*{exp['duration']}*")
                    st.write("---")
            else:
                st.warning("No work experience information extracted from your resume.")
    else:
        st.info("Please upload your resume and job description and run the analysis to see results here.")

# Recommendations Tab

with tab3:
    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations

        st.subheader("Recommendations to Improve Your Match")
        
        # Group recommendations by importance
        high_importance = []
        medium_importance = []
        low_importance = []
        info = []
        
        for rec in recommendations:
            if rec["importance"] == "high":
                high_importance.append(rec)
            elif rec["importance"] == "medium":
                medium_importance.append(rec)
            elif rec["importance"] == "low":
                low_importance.append(rec)
            else:
                info.append(rec)
        
        # Display high importance recommendations
        if high_importance:
            st.markdown("### Critical Improvements")
            for rec in high_importance:
                st.error(f"**{rec['category'].title()}**: {rec['recommendation']}")
        
        # Display medium importance recommendations
        if medium_importance:
            st.markdown("### Suggested Improvements")
            for rec in medium_importance:
                st.warning(f"**{rec['category'].title()}**: {rec['recommendation']}")
        
        # Display low importance recommendations
        if low_importance:
            st.markdown("### Minor Improvements")
            for rec in low_importance:
                st.info(f"**{rec['category'].title()}**: {rec['recommendation']}")
        
        # Display informational notes
        if info:
            st.markdown("### Notes")
            for rec in info:
                st.success(f"**{rec['category'].title()}**: {rec['recommendation']}")
        
        # Show resume and job description extracted text (in expanders)
        if st.session_state.resume_text and st.session_state.job_text:
            with st.expander("View Extracted Resume Text"):
                st.text_area("Resume Content", st.session_state.resume_text, height=300)
            
            with st.expander("View Extracted Job Description Text"):
                st.text_area("Job Description Content", st.session_state.job_text, height=300)
    else:
        st.info("Please upload your resume and job description and run the analysis to see recommendations here.")

# Footer

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Resume & Job Description Matcher </p>
    <p>Develop by Md Soumike Hassan</p>
    <p>Help by AI </p>
</div>
""", unsafe_allow_html=True)
📄 Resume-Job-Description Matcher
<p align="center"><i>An Intelligent NLP System to Evaluate Resume Fit with Job Descriptions</i></p> <p align="center"> <img src="https://img.shields.io/badge/NLP-Resume%20Matching-blueviolet" /> <img src="https://img.shields.io/badge/Python-3.8%2B-blue" /> <img src="https://img.shields.io/badge/License-MIT-green" /> </p>
🧭 Overview
Resume-Job-Description Matcher is an NLP-based tool that analyzes resumes and job descriptions to compute how well a candidate fits a job. It uses modern semantic matching techniques and keyword extraction to return a compatibility score and a skill match report.

✨ Features
<ul> <li>🔍 Intelligent skill and keyword extraction</li> <li>🤖 Semantic similarity computation with BERT, SBERT, TF-IDF</li> <li>📊 Match score & visualization of skill overlap</li> <li>✅ Matched vs. ❌ Missing skills analysis</li> <li>🌐 (Optional) Web-based interface using Streamlit</li> </ul>
🛠️ Technologies
<table> <tr><th>Technology</th><th>Purpose</th></tr> <tr><td>Python</td><td>Core Programming Language</td></tr> <tr><td>spaCy / NLTK</td><td>NLP Preprocessing</td></tr> <tr><td>scikit-learn</td><td>TF-IDF and ML Utilities</td></tr> <tr><td>Sentence-BERT</td><td>Semantic Similarity Matching</td></tr> <tr><td>Streamlit (Optional)</td><td>Interactive Web UI</td></tr> </table>
🧠 How It Works
<ol> <li>📥 Accepts resume and job description input (text or file)</li> <li>🧹 Cleans and preprocesses both documents</li> <li>🧠 Embeds text using NLP models (e.g., BERT/SBERT)</li> <li>📈 Computes a match score & extracts skills</li> <li>📤 Outputs results with scores and recommendations</li> </ol>

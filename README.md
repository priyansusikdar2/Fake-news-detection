<div align="center">

# 🛡️ Fake News Detection System

### *AI-Powered Truth Verification Engine*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Detect misinformation instantly with BERT-powered deep learning**

[Features](#✨-features) • [Demo](#🚀-quick-start) • [Architecture](#🏗️-system-architecture) • [API](#🎯-api-endpoints) • [Installation](#📦-installation)

</div>

---

## 📌 Overview

In an era of information overload, distinguishing between real and fake news is critical. This system leverages **state-of-the-art Natural Language Processing** to analyze news articles and determine their authenticity with high accuracy.

### Key Capabilities
- ⚡ **Real-time analysis** - Results in milliseconds
- 🎯 **92% accuracy** - Validated on 40,000+ articles
- 🔍 **Explainable AI** - Understand why content is flagged
- 🌐 **Web & API access** - Flexible integration options

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph Frontend["🌐 Frontend Layer"]
        UI[React/HTML5 Web Interface]
        API_REST[REST API Client]
    end

    subgraph Backend["⚙️ Backend Layer"]
        Flask[Flask Application Server]
        Router[Route Handler]
        
        subgraph Services["🔄 Core Services"]
            Preprocess[Text Preprocessor]
            Tokenizer[BERT Tokenizer]
            Predictor[Inference Engine]
            Analyzer[Rule-Based Analyzer]
        end
    end

    subgraph ML["🧠 Machine Learning Layer"]
        BERT[BERT/DistilBERT Model]
        Ensemble[Hybrid Decision Engine]
        Output[Classification Output<br/>FAKE / REAL]
    end

    subgraph Data["💾 Data Layer"]
        Models[(Trained Models<br/>*.pt files)]
        Cache[(Inference Cache)]
        Results[(Prediction History)]
    end

    UI --> API_REST
    API_REST --> Flask
    Flask --> Router
    Router --> Preprocess
    Preprocess --> Tokenizer
    Tokenizer --> Predictor
    Predictor --> BERT
    BERT --> Ensemble
    Analyzer --> Ensemble
    Ensemble --> Output
    Output --> Flask
    Flask --> API_REST
    API_REST --> UI
    
    BERT -.-> Models
    Predictor -.-> Cache
    Output -.-> Results

    style Frontend fill:#e1f5fe
    style Backend fill:#f3e5f5
    style ML fill:#e8f5e9
    style Data fill:#fff3e0
🔄 Prediction Pipeline
sequenceDiagram
    participant User as 👤 User
    participant UI as 🖥️ Web Interface
    participant API as 🔌 Flask API
    participant NLP as 📝 NLP Pipeline
    participant Model as 🧠 BERT Model
    participant Rules as 📏 Rule Engine

    User->>UI: Enter news article
    UI->>API: POST /api/predict
    API->>NLP: Clean & tokenize text
    NLP->>Model: Generate embeddings
    Model->>Model: Forward pass
    Model->>Ensemble: ML probability
    Rules->>Ensemble: Rule-based score
    Ensemble->>API: Final prediction (FAKE/REAL)
    API->>UI: JSON response
    UI->>User: Display result with confidence
📊 Detection Workflow
flowchart LR
    subgraph Input["📝 Input Processing"]
        A[Raw Text] --> B[Text Cleaning]
        B --> C[Tokenization]
        C --> D[BERT Encoding]
    end

    subgraph Analysis["🔍 Analysis Layer"]
        D --> E[ML Model]
        D --> F[Rule Engine]
        
        F --> G[Pattern Detection]
        F --> H[Clickbait Check]
        F --> I[Source Credibility]
        
        E --> J[Sensationalism Score]
        E --> K[Factual Language Score]
    end

    subgraph Decision["🎯 Decision Layer"]
        G --> L{Hybrid<br/>Ensemble}
        H --> L
        I --> L
        J --> L
        K --> L
        
        L --> M{Threshold > 0.48?}
        M -->|Yes| N[🔴 FAKE NEWS]
        M -->|No| O[🟢 REAL NEWS]
    end

    style Input fill:#e3f2fd
    style Analysis fill:#f3e5f5
    style Decision fill:#e8f5e9
✨ Features
🎯 Core Features
Feature	Description
Real-time Analysis	Instant predictions with millisecond latency
Confidence Scoring	Percentage-based reliability metrics
Risk Assessment	High/Medium/Low risk levels with color coding
Batch Processing	Analyze multiple articles simultaneously
API Access	RESTful endpoints for integration
🧠 Advanced Capabilities
Hybrid Detection - Combines ML with rule-based analysis

Clickbait Recognition - Identifies sensationalist patterns

Source Verification - Recognizes credible sources

Conspiracy Detection - Flags anti-establishment language

Technical Content Handling - Properly identifies neutral content

📊 Visualization Features
Confidence distribution charts

Probability bars (FAKE vs REAL)

Historical prediction tracking

Model performance metrics

🚀 Quick Start
Prerequisites
bash
# System Requirements
- Python 3.9 or higher
- 8GB RAM (recommended)
- 2GB free disk space
Installation (5 minutes)
bash
# 1. Clone the repository
git clone https://github.com/priyansusikdar2/Fake-news-detection.git
cd Fake-news-detection

# 2. Create virtual environment
python -m venv fake_news_env

# 3. Activate environment
# Windows:
fake_news_env\Scripts\activate
# Mac/Linux:
source fake_news_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download model files
python download_models.py

# 6. Run the application
python app/app.py
Open your browser to: http://localhost:5000

📦 Model Files
Due to GitHub's file size limitations, trained models are hosted on Google Drive:

File	Size	Description	Download
best_model.pt	255.9 MB	Highest accuracy model	🔗 Download
final_model.pt	255.9 MB	Final trained model	🔗 Download
bert_model.pt	417.7 MB	BERT backup model	🔗 Download
Auto-download: Run python download_models.py to download all models automatically.

🎯 API Endpoints
Base URL: http://localhost:5000/api
graph LR
    subgraph API["REST API"]
        A[POST /predict] --> B[Single Analysis]
        C[POST /predict/batch] --> D[Batch Analysis]
        E[GET /health] --> F[System Status]
        G[GET /model/info] --> H[Model Metadata]
        I[POST /calibrate] --> J[Adjust Threshold]
    end
Usage Examples
python
import requests

# Single prediction
response = requests.post('http://localhost:5000/api/predict', 
    json={'text': 'Your news article here'})

result = response.json()
print(f"📰 Result: {result['prediction']['label']}")
print(f"📊 Confidence: {result['prediction']['confidence']:.2%}")
print(f"⚠️ Risk Level: {result['prediction']['risk_level']}")

# Batch prediction
response = requests.post('http://localhost:5000/api/predict/batch',
    json={'texts': ['Article 1', 'Article 2', 'Article 3']})

for pred in response.json()['predictions']:
    print(f"{pred['label']}: {pred['confidence']:.2%}")
bash
# Using cURL
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"SHOCKING: Government hiding alien evidence!"}'
📊 Model Performance
graph LR
    subgraph Metrics["Performance Metrics"]
        A[Accuracy<br/>92%] 
        B[Precision<br/>91%]
        C[Recall<br/>93%]
        D[F1-Score<br/>92%]
        E[ROC-AUC<br/>0.96]
    end
Metric	Score	Description
Accuracy	92%	Overall correct predictions
Precision	91%	When predicting FAKE, how often correct
Recall	93%	FAKE articles correctly identified
F1-Score	92%	Harmonic mean of precision & recall
ROC-AUC	0.96	Model's ability to distinguish classes
🧪 Test Samples
✅ Real News (Should show REAL)
text
Scientists at Stanford University published a peer-reviewed study in the New England 
Journal of Medicine. The research tracked 10,000 participants over five years and found 
that regular exercise reduces heart disease risk by 30%. Lead researcher Dr. Sarah 
Johnson stated the findings are statistically significant.
❌ Fake News (Should show FAKE)
text
SHOCKING: Government hiding alien evidence from public! An anonymous Area 51 
whistleblower has leaked classified documents proving extraterrestrial contact! 
NASA officials have been lying for over 50 years! Wake up America!
🛠️ Technology Stack
mindmap
  root((Fake News Detection))
    Backend
      Flask
      Python 3.9+
      REST API
    Machine Learning
      PyTorch
      Transformers
      BERT/DistilBERT
      scikit-learn
    Frontend
      HTML5
      CSS3
      JavaScript
      Bootstrap 5
    Data Processing
      Pandas
      NumPy
      NLTK
    Visualization
      Matplotlib
      Seaborn
      Plotly
🔧 Configuration
Adjust Detection Threshold
bash
# Lower threshold = more FAKE detections (sensitive)
curl -X POST http://localhost:5000/api/calibrate \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.42}'

# Higher threshold = more REAL detections (conservative)
curl -X POST http://localhost:5000/api/calibrate \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.52}'
Threshold Guide
Threshold	Sensitivity	Use Case
0.40-0.44	Very High	Catch all potential fake news
0.45-0.48	High	General purpose (recommended)
0.49-0.52	Balanced	Default setting
0.53-0.56	Low	Minimize false positives
📁 Project Structure
text
fake-news-detection/
├── 📂 app/                    # Flask web application
│   ├── app.py                # Backend API & routes
│   ├── 📂 templates/         # HTML templates
│   └── 📂 static/            # CSS, JS files
├── 📂 src/                   # Core ML modules
│   ├── preprocessing.py      # Text cleaning
│   ├── dataset_loader.py     # Data utilities
│   ├── train.py             # Model training
│   ├── evaluate.py          # Performance metrics
│   └── predict.py           # Inference pipeline
├── 📂 notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb            # Data exploration
│   └── Training.ipynb       # Model training
├── 📂 model/                 # Trained models (download separately)
├── 📄 requirements.txt       # Python dependencies
├── 📄 download_models.py     # Model download script
└── 📄 README.md             # Documentation
🤝 Contributing
We welcome contributions! Please follow these steps:

flowchart LR
    A[Fork Repo] --> B[Create Branch]
    B --> C[Make Changes]
    C --> D[Run Tests]
    D --> E[Submit PR]
    E --> F[Review & Merge]
Fork the repository

Create feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Library/Tool	Purpose
🤗 Transformers	BERT model implementation
⚡ PyTorch	Deep learning framework
🌶️ Flask	Web framework
📊 scikit-learn	ML utilities
🎨 Bootstrap	UI components
📞 Contact & Support
<div align="center">
Priyansu Sikdar

https://img.shields.io/badge/GitHub-priyansusikdar2-181717?style=for-the-badge&logo=github
https://img.shields.io/badge/Email-Contact%2520Me-D14836?style=for-the-badge&logo=gmail

Project Link: https://github.com/priyansusikdar2/Fake-news-detection

</div>
<div align="center">
⭐ Show Your Support
If this project helped you, please consider giving it a star ⭐

Built with ❤️ for truth, transparency, and media literacy

Detecting misinformation, one article at a time.

</div> ```
How to Add This README
Copy the entire content above

Save it as README.md in your project root

Push to GitHub:

bash
git add README.md
git commit -m "Add comprehensive README with Mermaid architecture diagrams"
git push origin main

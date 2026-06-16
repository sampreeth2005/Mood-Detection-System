# 🧠 AI Mental Health Companion

An AI-powered academic project developed as part of a group project to provide basic emotional support and mood awareness using Facial Emotion Recognition and Sentiment Analysis.

> ⚠️ This application is for educational and research purposes only. It does not provide medical diagnosis, treatment, or professional mental health advice.

---

## 📖 Project Overview

AI Mental Health Companion is a Python-based application that combines computer vision and natural language processing techniques to analyze a user's emotional state.

The system uses:

* Facial emotion detection through webcam input
* Text sentiment analysis from user responses
* Mood tracking and history storage
* Personalized wellness recommendations

The goal is to demonstrate how Artificial Intelligence can be applied to emotional well-being and human-computer interaction.

---

## 🎯 Problem Statement

Mental health awareness is becoming increasingly important, but many individuals may not actively monitor their emotional well-being.

This project aims to:

* Detect emotional patterns
* Encourage self-reflection
* Provide basic supportive recommendations
* Demonstrate AI applications in mental health assistance

---

## 🚀 Features

### 😊 Facial Emotion Detection

* Captures user image through webcam
* Detects dominant facial emotion using DeepFace
* Supports emotions such as:

  * Happy
  * Sad
  * Angry
  * Fear
  * Surprise
  * Neutral

### 💬 Text-Based Mood Analysis

* Collects user responses through questionnaires
* Uses TextBlob sentiment analysis
* Classifies responses as:

  * Positive
  * Neutral
  * Negative

### 🧠 Hybrid Mood Classification

Combines:

* Facial emotion results
* Text sentiment results

To generate a final mood classification.

### 📊 Mood History Tracking

* Stores mood history locally in JSON format
* Maintains timestamps for each session
* Enables simple emotional pattern monitoring

### 💡 Recommendation Engine

Provides personalized suggestions based on detected mood:

* Relaxation techniques
* Healthy lifestyle recommendations
* Social support suggestions
* Professional help reminders when needed

### ⚠️ Pattern Recognition

Detects repeated negative moods and encourages seeking professional support.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Computer Vision

* OpenCV

### Facial Emotion Recognition

* DeepFace

### Natural Language Processing

* TextBlob

### Data Storage

* JSON

### Additional Libraries

* Datetime
* OS

---

## 📂 Project Structure

```text
AI-Mental-Health-Companion/
│
├── main.py
├── emotion_history.json
├── README.md
│
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AI-Mental-Health-Companion.git
```

```bash
cd AI-Mental-Health-Companion
```

### Install Dependencies

```bash
pip install opencv-python
pip install deepface
pip install textblob
```

### Download TextBlob Data

```bash
python -m textblob.download_corpora
```

---

## ▶️ Running the Project

```bash
python main.py
```

The application will:

1. Open webcam for facial emotion detection
2. Ask emotional well-being questions
3. Analyze mood
4. Generate recommendations
5. Save results locally

---

## 🧠 System Workflow

```text
User Input
      │
      ▼
Facial Emotion Detection
      │
      ▼
Text Sentiment Analysis
      │
      ▼
Mood Classification
      │
      ▼
Recommendation Engine
      │
      ▼
Store Emotion History
      │
      ▼
Pattern Analysis
```

---

## 📊 Sample Output

```text
AI Mental Health Companion

Facial Emotion Detected: happy

Text Mood Detected: positive

Final Mood Classification: positive

Great! You seem positive.

Keep up the good energy!
- Share positivity with others
- Continue productive habits
```

---

## 🎓 Academic Objectives

This project demonstrates concepts from:

* Artificial Intelligence
* Machine Learning Applications
* Computer Vision
* Natural Language Processing
* Human Computer Interaction
* Sentiment Analysis
* Emotion Recognition

---

## 🔮 Future Enhancements

* Voice emotion analysis
* AI chatbot integration
* Mental health dashboard
* Data visualization reports
* Multi-language support
* Cloud database integration
* Mobile application version
* Real-time counseling assistant

---

## 👥 Team Project

Developed as an academic group project.

### Team Members

* Bhogela Chetan Sai Sampreeth
* [Team Member Name]
* [Team Member Name]
* [Team Member Name]

(Add actual team member names)

---

## ⚠️ Disclaimer

This project is intended solely for educational and research purposes.

The system:

* Does not diagnose mental health conditions
* Does not replace professional medical advice
* Should not be used as a clinical tool

Users experiencing mental health concerns should consult qualified healthcare professionals.

---

## 📜 License

This project is developed for academic purposes.

---

## ⭐ Acknowledgements

* OpenCV
* DeepFace
* TextBlob
* Python Community
* VHACK 2.0 Hackathon Inspiration

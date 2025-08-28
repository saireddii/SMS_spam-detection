# SMS_spam-detection
📩 SMS Spam Detection using Machine Learning
This project was developed as part of my internship at CodSoft where I worked as a Machine Learning Intern. The goal was to build a robust machine learning model to classify SMS messages as spam or ham (not spam) using advanced Natural Language Processing (NLP) techniques.

🧠 Problem Statement
SMS spam is a growing problem affecting millions of users worldwide. The objective of this project is to create an intelligent classifier that can filter out spam messages, thereby improving user experience and security.

📊 Dataset
Source: UCI SMS Spam Collection Dataset

Total Messages: ~5,500

Classes:

ham: Legitimate messages

spam: Unwanted messages
🔧 Project Workflow
graph LR
    A[Data Preprocessing] --> B[Feature Extraction]
    B --> C[Model Building]
    C --> D[Evaluation]
    D --> E[Prediction & Deployment]
Step-by-Step Process
Data Preprocessing

Lowercased text

Removed punctuation and stopwords

Tokenization and stemming

Feature Extraction

TF-IDF Vectorizer to convert text into numerical features

Model Building

Trained models: Naive Bayes, SVM, Logistic Regression

Evaluation

Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Prediction & Deployment

Predict new/unseen messages

(Optional) Model serialization and deployment
Results
Accuracy: 0.9839 (98.39%)

Classification Report:

Class	Precision	Recall	F1-Score	Support
Ham	0.98	1.00	0.99	965
Spam	0.99	0.89	0.94	150
✅ Naive Bayes performed best due to its strength in handling text classification problems.

📽️ Demo Video
👉 [Attach your compressed spam_compressed.mp4 video here]

🧰 Technologies Used
Python

Pandas & NumPy

Scikit-learn

NLTK (Natural Language Toolkit)

Matplotlib & Seaborn

Jupyter Notebook

📌 Key Learnings
Preprocessing text data is crucial for accurate classification

Naive Bayes is highly effective for NLP tasks like spam detection

Evaluation metrics must go beyond accuracy in imbalanced datasets

🚀 How to Run
Clone this repository
git clone https://github.com/YerragudiChaitanya/SMS-_Spam_detection.git

Install dependencies
pip install -r requirements.txt

Run the notebook
Open sms_spam_detection (1).ipynb in Jupyter Notebook and run all cells.
💼 This project is a part of my internship at CodSoft under the role of Machine Learning Intern.
I’m always open to feedback and collaboration opportunities!


#SMSDetection #SpamClassifier #MachineLearning #CodSoftInternship #NLP #Python #MLProject #TextClassification #LinkedInProjects



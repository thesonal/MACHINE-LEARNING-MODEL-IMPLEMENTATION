# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SONAL YUVRAJ SONAWANE

*INTERN ID*: CTO4DH1793

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR


# 📧 Spam Detector ML Model

Detect spam emails using a hybrid approach that combines **natural language processing** with **custom feature engineering**. This model goes beyond text classification—it introduces a dynamic **Trust Scoring Mechanism** to rank email legitimacy.

---

## 🚀 Project Highlights

- 🔍 **Text + Metadata Fusion**: Combines TF-IDF vectorization of email content with numerical features like hyperlink density and sender reputation.
- 🧠 **Logistic Regression Pipeline**: A streamlined ML workflow with feature scaling and classification.
- 📊 **Evaluation Visuals**: Generates precision-recall metrics, a confusion matrix (`confusion_matrix.png`), and a ROC-AUC curve (`roc_auc_curve.png`).
- 🔐 **Trust Score Engine**: Intelligently adjusts spam probability based on sender reputation and content behavior.
- 🧪 **Simulated Data Framework**: Designed for educational use but easily adaptable to real-world datasets.

---

## 🧩 Features Engineered

| Feature             | Description                                                  |
|---------------------|--------------------------------------------------------------|
| Hyperlink Density   | Ratio of URLs to total word count in body text               |
| Sender Reputation   | Rule-based reputation score: Trusted, Spam, or Unknown       |
| TF-IDF Text Vectors | Captures keyword importance across email subject and body    |

---

## 📈 Model Performance (Simulated Dataset)

- ✅ **Accuracy**: 66.67%
- 🎯 **Precision**: 100.00%
- 🔁 **Recall**: 50.00%
- 📏 **F1 Score**: 66.67%
- 📉 **ROC AUC**: 0.8750

> 📂 Confusion matrix and ROC curve images are saved for visualization and debugging.

---

## 🔐 Trust Scoring Logic

Each email receives a **Trust Score (0–100)** based on:
- Spam probability from the ML model
- Sender reputation boosts or penalties

Examples:
- `trusted_sender@example.com` → Trust Score ≈ 93+
- `spam_sender@spam.com` → Trust Score ≈ 12
- `unknown_sender@random.com` → Trust Score ≈ 45–60

This score can help prioritize alerts or highlight suspicious senders in dashboards.

---

## ⚠️ Limitations

- 📦 **Synthetic Data**: Small, handcrafted dataset not suitable for production use.
- 🔐 **Reputation Rules**: Sender reputation uses fixed patterns; real-world systems need dynamic domain/IP blacklists.
- 🕵️‍♂️ **Evolving Threats**: Spammers adapt fast—models must retrain regularly.
- 📚 **Missing Context**: Email relevance may vary per user—personalized filters could improve accuracy.

---

## 🛠️ Tech Stack

- Python 🐍
- pandas, NumPy
- scikit-learn (TF-IDF, Logistic Regression)
- seaborn, matplotlib
- scipy (sparse matrix handling)

---

## OUTPUT

<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/bc96c247-41a3-4aab-845f-5701f39c4238" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/d4b3d90a-e18a-4cd1-b6a3-914283f55d41" />

# 📧 Spam Detector ML Model with Trust Scoring 🛡️

# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# 2. Simulate Sample Email Dataset
data = {
    'sender': [
        'trusted_sender@example.com', 'spam_sender@spam.com', 'marketing@example.com',
        'newsletter@trusted.com', 'unknown_sender@random.com', 'lottery_winner@spam.com',
        'friend@example.com', 'bank_alert@trusted.com', 'crypto_deal@spam.com',
        'meeting_invite@work.com'
    ],
    'subject': [
        'Project Update', "You've Won a Prize!", 'Exclusive Offer Just for You',
        'Weekly Newsletter', 'Important Account Information', 'Claim Your Free Lottery Winnings',
        'Catching Up', 'Security Alert: Suspicious Login', 'Hot New Cryptocurrency Investment',
        'Team Meeting Tomorrow'
    ],
    'body': [
        'Hi Team, here is the latest update... http://example.com/doc',
        'Congratulations! You won a car! http://spam.com/prize',
        "Limited-time offer: 50% off! http://example.com/offer",
        'Weekly news: http://trusted.com/newsletter',
        'Verify account: http://random.com/verify',
        'Provide bank details: http://spam.com/lottery',
        "Let's catch up soon!",
        'Suspicious login detected. http://trusted.com/security',
        'Crypto deal awaits! http://spam.com/crypto',
        'Meeting at 10 AM tomorrow.'
    ],
    'label': ['ham', 'spam', 'spam', 'ham', 'spam', 'spam', 'ham', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 3. Feature Engineering
def get_hyperlink_density(text):
    urls = re.findall(r"http[s]?://[^\s]+", text)
    word_count = len(text.split())
    return len(urls) / word_count if word_count > 0 else 0

def get_sender_reputation(sender):
    if 'trusted' in sender or 'work' in sender:
        return 1.0
    elif 'spam' in sender:
        return 0.0
    else:
        return 0.5

df['hyperlink_density'] = df['body'].apply(get_hyperlink_density)
df['sender_reputation'] = df['sender'].apply(get_sender_reputation)

# 4. Text Preprocessing
df['text'] = df['subject'] + ' ' + df['body']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X_text = tfidf_vectorizer.fit_transform(df['text'])

# 5. Combine Features
X_numerical = df[['hyperlink_density', 'sender_reputation']].values
X_combined = hstack([X_text, X_numerical])
y = df['label'].map({'ham': 0, 'spam': 1})

# 6. Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LogisticRegression(random_state=42))
])
pipeline.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Model Evaluation ---")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# 9. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_auc_curve.png')
plt.show()

# 10. Trust Score Function
def get_trust_score(row, model, vectorizer):
    text = row['subject'] + ' ' + row['body']
    hyperlink_density = get_hyperlink_density(row['body'])
    sender_reputation = get_sender_reputation(row['sender'])

    text_features = vectorizer.transform([text])
    numerical_features = np.array([[hyperlink_density, sender_reputation]])
    combined_features = hstack([text_features, numerical_features])
    spam_prob = model.predict_proba(combined_features)[:, 1][0]

    trust = (1 - spam_prob) * 100
    if sender_reputation == 0.0:
        trust *= 0.5
    elif sender_reputation == 1.0:
        trust = min(100, trust * 1.2)
    return trust

df['trust_score'] = df.apply(lambda row: get_trust_score(row, pipeline, tfidf_vectorizer), axis=1)
print("\n--- Trust Scores ---")
print(df[['sender', 'subject', 'label', 'trust_score']])

# 11. Model Limitations
print("""
--- Model Limitations ---
- Dataset is synthetic and small; real-world spam is far more varied.
- Sender reputation system is manually crafted and static.
- Trust scoring is rule-based and would need dynamic tuning for production.
- Feature set is basic: real systems benefit from richer NLP and behavioral signals.
- Spam intent varies per recipient; personalized detection would improve accuracy.
""")
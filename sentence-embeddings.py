import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


np.random.seed(42)
descriptions = [
    "Implement multi-factor authentication",
    "Update firewall rules",
    "Replace legacy systems",
    "Establish an incident response team",
    "Train employees on phishing awareness",
    "Conduct quarterly vulnerability assessments",
    "Submit cybersecurity audit report",
    "Review vendor risk policies",
    "Fix misconfigured cloud storage",
    "Patch known vulnerabilities",
    "Implement encryption across endpoints",
    "Delay in vendor onboarding process",
    "Outdated software caused breach",
    "Unauthorized access detected",
    "Failure to backup critical data",
    "Phishing attack compromised credentials"
] * 10  # expand the dataset

days_to_resolve = np.random.choice([90, 200, 400], size=len(descriptions), p=[0.4, 0.35, 0.25])


df = pd.DataFrame({
    'issue_description': descriptions,
    'days_to_resolve': days_to_resolve
})


def categorize_days(days):
    if days < 180:
        return 0
    elif days <= 365:
        return 1
    else:
        return 2


df['label'] = df['days_to_resolve'].apply(categorize_days)

model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(df['issue_description'].tolist())
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


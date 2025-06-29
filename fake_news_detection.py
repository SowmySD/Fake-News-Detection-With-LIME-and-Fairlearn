import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop rows with null text
df.dropna(subset=['text'], inplace=True)

def clean_text(text):
  text = text.lower()  # lowercase
  text = re.sub(r'\d+', '', text)  # remove numbers
  text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
  text = text.strip()  # remove whitespace
  text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
  return text

# Apply preprocessing
df['clean_text'] = df['text'].apply(clean_text)

# Prepare tokenizer input
texts = df['clean_text'].tolist()
labels = df['label'].values

vocab_size = 10000
max_length = 500
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
    ])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=7, validation_data=(X_test, y_test), batch_size=64)

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

pip install lime

import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])

# Define prediction function for LIME
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    predictions = model.predict(padded_seq)
    return np.hstack((1 - predictions, predictions))  # shape [n, 2]

i = 42  # pick an example index
sample_text = df.iloc[i]['text']
true_label = df.iloc[i]['label']

print(f"\nTrue Label: {'Fake' if true_label == 0 else 'Real'}")
print(f"Text Snippet: {sample_text[:500]}...\n")

exp = explainer.explain_instance(sample_text, predict_proba, num_features=10, top_labels=1)

exp.show_in_notebook(text=True)

# Get LIME's predicted class
pred_label = exp.top_labels[0]

# Visualize the explanation
fig = exp.as_pyplot_figure(label=pred_label)
plt.title(f'LIME Explanation (Predicted: {"Fake" if pred_label == 0 else "Real"})')
plt.tight_layout()
plt.show()

# Table of word importance values
word_weights = exp.as_list(label=pred_label)
importance_df = pd.DataFrame(word_weights, columns=['Word', 'Weight'])
importance_df['Contribution'] = importance_df['Weight'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

print("\n🔍 Top Word Contributions to Prediction:\n")
print(importance_df)

"""**BIAS DETECTION**"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# df = combined and cleaned Fake + Real dataset
# Preprocess texts
texts = df['text']
sequences = tokenizer.texts_to_sequences(texts)
padded_texts = pad_sequences(sequences, maxlen=max_length)

# Predictions
y_true = df['label'].values
y_pred_probs = model.predict(padded_texts)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

subject_groups = df['subject'].unique()
bias_subject = []

for subject in subject_groups:
  mask = df['subject'] == subject
  acc = accuracy_score(y_true[mask], y_pred[mask])
  prec = precision_score(y_true[mask], y_pred[mask])
  rec = recall_score(y_true[mask], y_pred[mask])
  f1 = f1_score(y_true[mask], y_pred[mask])
  bias_subject.append((subject, acc, prec, rec, f1))

bias_subject_df = pd.DataFrame(bias_subject, columns=['Subject', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print("\n📈 Performance by Subject:")
print(bias_subject_df.sort_values(by='F1-score'))

df['text_len'] = df['text'].apply(lambda x: len(x.split()))
df['len_group'] = pd.cut(df['text_len'], bins=[0, 300, 600, 1000, 5000], labels=['Short', 'Medium', 'Long', 'Very Long'])

bias_length = []
for group in df['len_group'].unique():
  mask = df['len_group'] == group
  acc = accuracy_score(y_true[mask], y_pred[mask])
  f1 = f1_score(y_true[mask], y_pred[mask])
  bias_length.append((group, acc, f1))

bias_len_df = pd.DataFrame(bias_length, columns=['Length Group', 'Accuracy', 'F1-score'])
print("\n📏 Performance by Text Length:")
print(bias_len_df.sort_values(by='F1-score'))

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year_group'] = pd.cut(df['date'].dt.year, bins=[2015, 2017, 2019, 2021, 2023], labels=['2016-17', '2018-19', '2020-21', '2022-23'])

bias_date = []
for year in df['year_group'].dropna().unique():
  mask = df['year_group'] == year
  acc = accuracy_score(y_true[mask], y_pred[mask])
  f1 = f1_score(y_true[mask], y_pred[mask])
  bias_date.append((year, acc, f1))

bias_date_df = pd.DataFrame(bias_date, columns=['Year Range', 'Accuracy', 'F1-score'])
print("\n📅 Performance by Article Date:")
print(bias_date_df.sort_values(by='F1-score'))

import seaborn as sns
import matplotlib.pyplot as plt

# Subject Bias Plot
plt.figure(figsize=(10, 5))
sns.barplot(x='F1-score', y='Subject', data=bias_subject_df.sort_values(by='F1-score'))
plt.title('F1-score by News Subject')
plt.show()

# Length Bias Plot
plt.figure(figsize=(6, 4))
sns.barplot(x='Length Group', y='F1-score', data=bias_len_df)
plt.title('F1-score by Text Length Group')
plt.show()

"""**A. Ethical Data Gathering**"""

# Count samples by subject
subject_counts = df['subject'].value_counts()
print(subject_counts)

# Visualize imbalance
subject_counts.plot(kind='bar', title="Subject Distribution", figsize=(10, 4))
plt.show()

"""**B. Ethical Data Preprocessing**"""

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Vectorize text again for fairness modeling
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(df['text'])
y = df['label']

# Oversample minority subject groups (optional: by `subject`)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_vec, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"""**C. Ethical Modeling**"""

pip install fairlearn

from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, equalized_odds_difference, selection_rate

# Convert sparse matrix to dense
X_dense = X_vec.toarray()

# Re-sample and split
X_resampled, y_resampled = ros.fit_resample(X_dense, y)
# Prepare sensitive attribute before resampling
A = df['subject'].reset_index(drop=True)


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_dense, y)
A_resampled, _ = ros.fit_resample(A.to_frame(), y)  # Resample sensitive attribute
A_resampled = A_resampled['subject']  # Extract as Series

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_resampled, y_resampled, A_resampled, test_size=0.2, random_state=42)

# Train fairness-aware model
fair_model = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=EqualizedOdds(), eps=0.01)
fair_model.fit(X_train, y_train, sensitive_features=A_train)

# Predictions
y_pred_fair = fair_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred_fair)
print("✅ Fair Model Accuracy:", acc)

# Evaluate fairness metrics
mf = MetricFrame(metrics={'accuracy': accuracy_score,
    'selection_rate': selection_rate},
    y_true=y_test,
    y_pred=y_pred_fair,
    sensitive_features=A_test)

print("\nFairness Metrics by Subject:\n", mf.by_group)
print("\nEqualized Odds Difference:", equalized_odds_difference(y_test, y_pred_fair, sensitive_features=A_test))

import matplotlib.pyplot as plt
import seaborn as sns

# Fairness metric plot
plt.figure(figsize=(10, 4))
sns.barplot(x=mf.by_group.index, y=mf.by_group['accuracy'])
plt.xticks(rotation=45)
plt.title('Accuracy by Subject Group (Fair Model)')
plt.show()


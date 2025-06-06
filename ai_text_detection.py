import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


#%% 1. Dataset Analysis

df = pd.read_csv('C:/Users/flora/project_ai/data/train_essays.csv')

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['generated'].value_counts(normalize=True))

df['text_length'] = df['text'].apply(len)
print("\nText Length Statistics:")
print(df.groupby('generated')['text_length'].describe())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='generated', y='text_length', data=df)
plt.title('Text Length by Class')

plt.subplot(1, 2, 2)
df['text_length'].hist(by=df['generated'], bins=50, alpha=0.7)
plt.suptitle('Text Length Distribution')
plt.tight_layout()
plt.show()


#%% 2. Text Processing and Transformation

nltk.download(['punkt', 'stopwords', 'wordnet'])

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
bow = CountVectorizer(max_features=5000, ngram_range=(1, 2))

X_tfidf = tfidf.fit_transform(df['processed_text'])
X_bow = bow.fit_transform(df['processed_text'])


#%% 3. Application of Different Embedding Techniques

# Word2Vec model
sentences = [text.split() for text in df['processed_text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

def get_w2v_embeddings(texts, model):
    embeddings = []
    for text in texts:
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)

X_w2v = get_w2v_embeddings(df['processed_text'], w2v_model)

# Doc2Vec model
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) 
               for i, text in enumerate(df['processed_text'])]
d2v_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=2, workers=4)

X_d2v = np.array([d2v_model.infer_vector(text.split()) 
                 for text in df['processed_text']])

# BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts.tolist(), return_tensors='pt', 
                      padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

sample_size = 300
X_bert = get_bert_embeddings(df['processed_text'].iloc[:sample_size])

def evaluate_multiple_models(X, y, name=''):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel='linear', probability=True),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,))
    }

    print(f"\n--- {name} Embedding Evaluation ---")
    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {name} embeddings...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            'Embedding': name,
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1': report['weighted avg']['f1-score']
        })

        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix ({name})')
        plt.show()

    return results


#%% 4. Clustering and/or Classification on Embedded Data

X = X_tfidf
y = df['generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,))
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1': report['weighted avg']['f1-score']
    })

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)


#%% 5. Results Analysis and Visualization

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_w2v)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_w2v)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], 
                hue=df['generated'], style=clusters,
                palette='viridis', alpha=0.7)
plt.title('t-SNE Visualization with Clusters')
plt.show()
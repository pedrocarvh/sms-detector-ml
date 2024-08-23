markdown
# Projeto de Previsão de Spam em SMS

Este projeto visa construir um modelo de Machine Learning para classificar mensagens de texto como "spam" ou "ham" (não spam). Utilizamos um conjunto de dados com mensagens SMS rotuladas para treinar e testar o modelo. Este documento fornece um guia completo sobre o carregamento dos dados, pré-processamento, construção do modelo e avaliação.

## Índice

- [Introdução](#introdução)
- [Pré-requisitos](#pré-requisitos)
- [Carregamento e Limpeza dos Dados](#carregamento-e-limpeza-dos-dados)
- [Análise Exploratória dos Dados](#análise-exploratória-dos-dados)
- [Processamento de Texto](#processamento-de-texto)
- [Construção do Modelo](#construção-do-modelo)
- [Resultados e Avaliação](#resultados-e-avaliação)
- [Conclusão](#conclusão)

## Introdução

O objetivo deste projeto é desenvolver um modelo que classifica mensagens SMS em duas categorias: "spam" e "ham". O processo inclui o pré-processamento dos dados, a construção do modelo e a avaliação de seu desempenho.

## Pré-requisitos

Antes de começar, você precisará das seguintes ferramentas e bibliotecas instaladas:

- **Python**: Linguagem de programação para ciência de dados e Machine Learning.
- **Bibliotecas**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `scikit-learn`
  - `imblearn`

Instale as bibliotecas necessárias com:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn imbalanced-learn
```


## Carregamento e Limpeza dos Dados

### Carregamento dos Dados

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/projeto-ml/sms/spam.csv', encoding='latin-1')


### Visualização Inicial

```python
df.sample(5)
df.shape
```

### Limpeza dos Dados

```python
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
```

### Transformação de Rótulos

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
```

### Verificação de Dados

```python
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates(keep='first')
```

## Análise Exploratória dos Dados

### Distribuição de Classes

```python
import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
plt.show()
```

### Características das Mensagens

```python
import nltk
nltk.download('punkt')
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
```

### Matriz de Correlação

```python
df_numeric = df.drop(columns=['text'])
corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True)
plt.show()
```

## Processamento de Texto

### Preparação do Texto

```python
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

df['transformed_text'] = df['text'].apply(transform_text)
```

### Visualização das Palavras

```python
from wordcloud import WordCloud

wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
plt.show()

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
plt.show()
```

### Distribuição de Palavras

```python
from collections import Counter

spam_corpus = [word for msg in df[df['target'] == 1]['transformed_text'].tolist() for word in msg.split()]
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

ham_corpus = [word for msg in df[df['target'] == 0]['transformed_text'].tolist() for word in msg.split()]
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
```

## Construção do Modelo

### Preparação dos Dados

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer(max_features=3000)
tfidf = TfidfVectorizer(max_features=3000)

X_cv = cv.fit_transform(df['transformed_text']).toarray()
X_tfidf = tfidf.fit_transform(df['transformed_text']).toarray()
X = np.hstack((X_cv, X_tfidf))
y = df['target'].values
```

### Pipeline com SMOTE e Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from imblearn.pipeline import make_pipeline as imblearn_make_pipeline
from imblearn.over_sampling import SMOTE

stratified_kfold = StratifiedKFold(n_splits=5)

pipeline = imblearn_make_pipeline(
    SMOTE(random_state=2),
    MultinomialNB()
)

cv_scores = cross_val_score(pipeline, X, y, cv=stratified_kfold, scoring='accuracy')
print("Validação Cruzada Estratificada - Accuracy média:", cv_scores.mean())
print("Validação Cruzada Estratificada - Desvio Padrão:", cv_scores.std())
```

### Ajuste de Hiperparâmetros

```python
param_grid = {
    'smote__sampling_strategy': [0.5, 1.0, 'auto'],
    'multinomialnb__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=stratified_kfold,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)
print("Melhor parâmetro alpha:", grid_search.best_params_['multinomialnb__alpha'])
print("Melhor parâmetro sampling_strategy:", grid_search.best_params_['smote__sampling_strategy'])
```

### Avaliação do Modelo

```python
from sklearn.metrics import classification_report

y_pred = grid_search.predict(X)
print(classification_report(y, y_pred))
```

## Resultados e Avaliação

### Métricas de Desempenho

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
```

### Curva ROC

```python
from sklearn.metrics import roc_curve, auc

y_prob = grid_search.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

## Conclusão

O modelo desenvolvido demonstrou um desempenho robusto na classificação de mensagens SMS como spam ou ham. O uso de técnicas como SMOTE e validação cruzada ajudou a melhorar a performance geral. As métricas indicam que o modelo é eficiente na identificação de spam, oferecendo uma solução eficaz para a filtragem de mensagens.

---

Para mais informações, contribuições ou dúvidas, sinta-se à vontade para abrir uma issue ou enviar um pull request.

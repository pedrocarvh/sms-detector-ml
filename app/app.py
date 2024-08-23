import pickle
import streamlit as st
import numpy as np
import os
import time

# Caminhos dos arquivos
base_path = 'app'
model_path = os.path.join(base_path, 'spam_classifier_model.pkl')
count_vectorizer_path = os.path.join(base_path, 'count_vectorizer.pkl')
tfidf_vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

# Verificar se os arquivos existem
for path in [model_path, count_vectorizer_path, tfidf_vectorizer_path]:
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        st.write(f"Arquivos no diretório {base_path}: {os.listdir(base_path)}")
        st.stop()

# Carregar os arquivos
with open(model_path, 'rb') as model_file:
    best_mnb = pickle.load(model_file)

with open(count_vectorizer_path, 'rb') as count_vectorizer_file:
    count_vectorizer = pickle.load(count_vectorizer_file)

with open(tfidf_vectorizer_path, 'rb') as tfidf_vectorizer_file:
    tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

# Código do Streamlit
st.title('Verificador de Spam de SMS')

# Adicionar CSS para estilizar a página
st.markdown("""
<style>
    .title {
        font-size: 2em;
        color: #4CAF50;
    }
    .message-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .result-box {
        font-size: 2em;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        transition: transform 0.5s ease, opacity 0.5s ease;
    }
    .spam {
        background: linear-gradient(45deg, #f44336, #e57373);
        animation: slide-in 1s;
    }
    .not-spam {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        animation: slide-in 1s;
    }
    @keyframes slide-in {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    .history-box {
        margin-top: 20px;
        padding: 10px;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .tip-box {
        margin-top: 20px;
        padding: 10px;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        background-color: #e9fbe9;
    }
</style>
""", unsafe_allow_html=True)

user_input = st.text_area('Digite uma mensagem SMS:', '', key='input_text')

# Armazenar histórico na sessão
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button('Classificar'):
    if user_input:
        with st.spinner('Classificando...'):
            time.sleep(1)  # Simula um pequeno delay para a animação
            # Vetorização das características
            X_cv = count_vectorizer.transform([user_input])
            X_tfidf = tfidf_vectorizer.transform([user_input])
            
            # Combinar as características
            X_combined = np.hstack((X_cv.toarray(), X_tfidf.toarray()))
            
            # Previsão
            prediction = best_mnb.predict(X_combined)
            
            # Adicionar ao histórico
            result = 'Spam' if prediction[0] == 1 else 'Não Spam'
            st.session_state.history.append({'message': user_input, 'result': result})
            
            # Exibir resultado com estilo
            if prediction[0] == 1:
                st.markdown('<div class="result-box spam">🚩 Spam</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box not-spam">✅ Não Spam</div>', unsafe_allow_html=True)

# Mostrar histórico
if st.session_state.history:
    st.subheader('Histórico de Mensagens')
    for entry in st.session_state.history:
        st.write(f"Mensagem: {entry['message']}")
        st.write(f"Resultado: {entry['result']}")
        st.write('---')

# Dicas sobre Spam
st.subheader('Dicas para Identificar Spam')
st.markdown("""
Procure por mensagens que solicitam informações pessoais.
Desconfie de ofertas que parecem boas demais para ser verdade.
Verifique se há erros gramaticais ou ortográficos na mensagem.
""")

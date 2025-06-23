import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import tempfile

# --- O CORRETO É COLOCAR st.set_page_config() AQUI, ANTES DE QUALQUER OUTRO COMANDO STREAMLIT ---
st.set_page_config(layout="wide", page_title="PDF Q&A com Ollama")

# --- Configurações de Modelo ---
MODEL_EMBEDDING = 'sentence-transformers/all-MiniLM-L6-v2'
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_LLM = "qwen2.5"

# --- Carregar o modelo de embeddings (com cache) ---
@st.cache_resource
def carregar_modelo_embedding():
    return SentenceTransformer(MODEL_EMBEDDING)

embedding_model = carregar_modelo_embedding()

# --- Função para extrair texto de PDFs ---
def extrair_texto_pdf(file):
    texto = ""
    try:
        leitor = PyPDF2.PdfReader(file)
        for pagina in leitor.pages:
            texto += pagina.extract_text() + "\n"
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}")
        return ""
    return texto

# --- Interface Streamlit ---
st.title("📚 Perguntas com PDFs usando Embeddings + LLM (Ollama)")

st.markdown("""
    Este aplicativo permite que você faça upload de **arquivos PDF** e, em seguida, insira uma **pergunta** para obter uma resposta baseada no conteúdo dos documentos.
    Utilizamos modelos de **embeddings** para encontrar o documento mais relevante e um **Large Language Model (LLM)** via Ollama para gerar a resposta.
""")

st.divider() # Adiciona uma linha divisória

# --- Seção de Upload e Pergunta ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Envie seus PDFs")
    uploaded_files = st.file_uploader(
        "Arraste e solte ou clique para enviar",
        type="pdf",
        accept_multiple_files=True,
        help="Faça upload de um ou mais arquivos PDF para análise."
    )

with col2:
    st.subheader("2. Digite sua Pergunta")
    pergunta = st.text_input(
        "Qual a sua pergunta sobre os documentos?",
        placeholder="Ex: Qual o principal objetivo do documento X?",
        help="Insira a pergunta que você deseja responder com base nos PDFs enviados."
    )

st.divider()

# --- Botão de Consulta ---
if st.button("🚀 Consultar PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Por favor, envie pelo menos um arquivo PDF para consultar.")
    elif not pergunta:
        st.warning("Por favor, digite sua pergunta.")
    else:
        with st.spinner("Processando documentos e consultando o modelo..."):
            # Extrair textos dos PDFs
            documentos = []
            nomes_arquivos = []
            for file in uploaded_files:
                texto_extraido = extrair_texto_pdf(file)
                if texto_extraido: # Apenas adiciona se a extração foi bem-sucedida
                    documentos.append(texto_extraido)
                    nomes_arquivos.append(file.name)
            
            if not documentos:
                st.error("Não foi possível extrair texto de nenhum dos PDFs enviados. Verifique se os arquivos estão corretos.")
                # st.stop() # st.stop() pode ser muito agressivo, use se for essencial parar a execução.
                           # Neste caso, a mensagem de erro já é suficiente.

            # Calcular embeddings
            embeddings_documentos = embedding_model.encode(documentos)
            embedding_pergunta = embedding_model.encode([pergunta])[0]

            # Calcular similaridades
            similaridades = cosine_similarity([embedding_pergunta], embeddings_documentos)[0]
            max_index = similaridades.argmax()
            threshold = 0.3 # Limiar de similaridade para considerar um documento relevante

            st.subheader("📊 Análise de Similaridade:")
            # Usar st.expander para mostrar os detalhes de similaridade
            with st.expander("Ver similaridade de cada documento com a pergunta"):
                for i, sim in enumerate(similaridades):
                    st.markdown(f"📄 **{nomes_arquivos[i]}** → Similaridade: `{sim:.2f}`")
                
                st.info(f"O documento mais similar é: **{nomes_arquivos[max_index]}** com similaridade de `{similaridades[max_index]:.2f}`")

            if similaridades[max_index] > threshold:
                documento_relevante = documentos[max_index]

                # Construir prompt para o LLM
                prompt = f"""<|im_start|>system
Você é um assistente que responde perguntas baseado exclusivamente no contexto fornecido.
<|im_end|>
<|im_start|>context
{documento_relevante}
<|im_end|>
<|im_start|>user
{pergunta}<|im_end|>
<|im_start|>assistant
"""

                payload = {
                    "model": MODEL_LLM,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                }

                try:
                    resposta = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # Adiciona timeout
                    resposta.raise_for_status() # Lança exceção para status de erro HTTP
                    conteudo = resposta.json().get("response", "").split("<|im_end|>")[0].strip()

                    st.subheader("💡 Resposta do Modelo LLM:")
                    st.markdown(f"<p style='background-color:#e0f7fa; padding:15px; border-radius:8px;'>{conteudo}</p>", unsafe_allow_html=True)
                    st.info(f"Resposta baseada no documento: **{nomes_arquivos[max_index]}**")

                except requests.exceptions.Timeout:
                    st.error("A requisição para o Ollama excedeu o tempo limite. Verifique se o servidor Ollama está rodando e acessível.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Erro ao comunicar com o servidor Ollama: {e}")
                except Exception as e:
                    st.error(f"Ocorreu um erro inesperado: {e}")
            else:
                st.warning("Nenhum documento foi considerado relevante o suficiente (similaridade abaixo de 0.3) para responder à sua pergunta.")
                st.info("Tente refinar sua pergunta ou enviar documentos mais relevantes.")

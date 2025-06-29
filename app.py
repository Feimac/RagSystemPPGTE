import os
import json
import time
import re
from dotenv import load_dotenv

import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Tenta importar Faiss
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False

# Carrega variáveis de ambiente
load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://10.20.50.50:5000/api/generate")
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_LLM       = os.getenv("MODEL_LLM", "llama3.2")

# Configurações fixas
JSON_PATHS  = ["regulamento_ppgte_corrigido.json", "selecao.json"]
TOP_K       = 4
THRESHOLD   = 0.35
TEMPERATURE = 0.3
TOP_P       = 0.9

# Mapeamento de headings para nomes amigáveis
FRIENDLY_NAMES = {
    # Regulamento
    "CAPÍTULO II – DO CORPO DOCENTE": "Corpo Docente",
    "REGIME ACADÊMICO": "Requisitos Acadêmicos",
    "CAPÍTULO V": "Comissão Examinadora",
    "DISPOSIÇÕES GERAIS E TRANSITÓRIAS": "Disposições Gerais",
    # Seleção (exemplos)
    "Art. 1º": "Seleção: Art. 1º",
    "Art. 2º": "Seleção: Art. 2º",
    # … adicione os demais artigos conforme o JSON de seleção …
}

def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9 ]", "", text).strip().lower()
    return s.replace(" ", "-")

# Streamlit page config
st.set_page_config(layout="wide", page_title="Dúvidas e Respostas PPGTE")

# Cache do modelo de embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_EMBEDDING)

# Carrega todas as seções de ambos os JSONs
def load_sections():
    all_secs = []
    for path in JSON_PATHS:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        title = doc.get("title", "")
        if title:
            all_secs.append({"heading": os.path.basename(path).replace(".json",""), "content": title})
        for sec in doc.get("sections", []):
            all_secs.append(sec)
    return all_secs

# Gera e normaliza embeddings das seções
@st.cache_data
def load_section_embeddings(_model, sections):
    texts = [f"## {sec['heading']}\n{sec['content']}" for sec in sections]
    embs = np.vstack([_model.encode(t).astype('float32') for t in texts])
    if USE_FAISS:
        faiss.normalize_L2(embs)
    else:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms == 0, 1, norms)
    return embs

# Constrói índice FAISS
def build_faiss_index(embs):
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

# Chama API Ollama com retry
def call_ollama(payload: dict, attempts: int = 3):
    for i in range(attempts):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if i < attempts - 1:
                time.sleep(2)
            else:
                raise

# Interface
st.title("❓ Dúvidas e Respostas PPGTE")
st.markdown("Pergunte sobre o documento e receba respostas com citações interativas de ambos os JSONs.")

question = st.text_input("Digite sua pergunta:")

if st.button("🚀 Perguntar"):
    if not question:
        st.warning("Por favor, insira uma pergunta.")
    else:
        # carrega modelo, seções e embeddings
        model    = load_embedding_model()
        sections = load_sections()
        embs     = load_section_embeddings(model, sections)

        # embedding da query
        q_emb = model.encode(question).astype('float32')
        if USE_FAISS:
            faiss.normalize_L2(q_emb.reshape(1, -1))
            index = build_faiss_index(embs)
            sims, idxs = index.search(q_emb.reshape(1, -1), TOP_K)
            sims, idxs = sims[0], idxs[0]
        else:
            sims  = cosine_similarity([q_emb], embs)[0]
            idxs  = sims.argsort()[::-1][:TOP_K]

        # exibe top-K
        st.subheader("🔍 Seções Encontradas")
        for sim, idx in zip(sims, idxs):
            sec      = sections[idx]
            friendly = FRIENDLY_NAMES.get(sec["heading"], sec["heading"])
            slug     = slugify(friendly)
            st.markdown(f"<a id='{slug}'></a>", unsafe_allow_html=True)
            with st.expander(f"{friendly} (sim={sim:.2f})"):
                st.write(sec["content"])

        # threshold usando o melhor score
        best_sim = sims[0]
        if best_sim < THRESHOLD:
            st.warning("Nenhuma seção atingiu similaridade mínima.")
        else:
            # monta contexto
            context = "\n\n".join(
                f"<section name=\"{FRIENDLY_NAMES.get(sections[i]['heading'], sections[i]['heading'])}\">\n"
                f"{sections[i]['content']}\n</section>"
                for i in idxs
            )

            system_msg = (
                "Você é um assistente que responde apenas com base nos documentos JSON. "
                "Cada seção tem um título (‘heading’) e conteúdo (‘content’). "
                "Cite ao final de cada informação **[Seção: Nome]**."
            )
            prompt = (
                f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
                f"<|im_start|>context\n{context}\n<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant"
            )
            payload = {
                "model":   MODEL_LLM,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": TEMPERATURE, "top_p": TOP_P}
            }

            try:
                res    = call_ollama(payload)
                answer = res.get("response", "").split("<|im_end|>")[0].strip()

                highlighted = re.sub(r"(\[Seção: .*?\])", r"**\1**", answer)
                st.subheader("💡 Resposta com Citações e Destaques")
                st.markdown(
                    f"<div style='background:#e0f7fa;padding:15px;border-radius:8px'>{highlighted}</div>",
                    unsafe_allow_html=True
                )

                cites = re.findall(r"\[Seção: ([^\]]+)\]", answer)
                if cites:
                    st.subheader("🔗 Referências")
                    for sec_name in dict.fromkeys(cites):
                        slug = slugify(sec_name)
                        st.markdown(f"- [{sec_name}](#{slug})")

            except Exception as e:
                st.error(f"Erro na API Ollama: {e}")

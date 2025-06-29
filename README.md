
# 📖 RagSystemPPGTE

Um sistema de **RAG (Retrieval-Augmented Generation)** para responder automaticamente perguntas sobre o Regulamento Interno do Programa de Pós-Graduação em Tecnologia e Sociedade (UTFPR), usando Streamlit e embeddings de texto.

---

## 🚀 Funcionalidades

- **Upload** do JSON do regulamento e (opcional) de outros documentos (ex.: `selecao.json`).  
- **Indexação** das seções do documento via embeddings (MiniLM) e Faiss (ou similaridade por cosseno).  
- **Busca** de trechos relevantes com base em similaridade semântica.  
- **Geração de respostas** pelo LLM (via Ollama API), citando seções e criando links internos.  
- **Interface web** interativa com Streamlit.

---

## 🛠️ Pré-requisitos

- Python 3.8+  
- [pipenv](https://pipenv.pypa.io/) ou [virtualenv](https://docs.python.org/3/library/venv.html)  
- Acesso à API Ollama (ou outro LLM compatível)  
- (Opcional) Faiss para acelerar busca por similaridade

---

## 🔧 Instalação

1. Clone o repositório:  
   ```bash
   git clone git@github.com:Feimac/RagSystemPPGTE.git
   cd RagSystemPPGTE
````

2. Crie e ative o ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure variáveis de ambiente (copie e edite `.env`):

   ```ini
   OLLAMA_API_URL=http://<seu-endereco-ollama>/api/generate
   MODEL_EMBEDDING=sentence-transformers/all-MiniLM-L6-v2
   MODEL_LLM=llama3.2
   ```

---

## 🚩 Uso

Execute a aplicação Streamlit:

```bash
streamlit run app.py
```

* Abra o navegador em `http://localhost:8501`.
* Faça perguntas sobre o regulamento e obtenha respostas com citações e links internos.
* Para incluir outro JSON (ex.: `selecao.json`), basta colocar na pasta e ajustar `JSON_PATH` em `app.py`.

---

## 🧩 Estrutura do projeto

```
RagSystemPPGTE/
├── app.py                  # Aplicação Streamlit principal
├── requirements.txt        # Dependências Python
├── regulament_ppgte_corrigido.json  # JSON do regulamento
├── selecao.json            # Outro JSON (opcional)
├── pdfs/                   # PDFs fonte (Regulamento, seleção etc.)
├── .env                    # Variáveis de ambiente
├── README.md               # Este arquivo
└── ...                     # outros módulos/logs
```

---

## ⚙️ Personalização

* **Ajustar `TOP_K`, `THRESHOLD` ou `FRIENDLY_NAMES`** em `app.py` para melhorar a recuperação de seções.
* **Habilitar Faiss** instalando com `pip install faiss-cpu` e colocando `USE_FAISS = True`.
* **Customizar prompt** no bloco `system_msg` para obedecer ao seu estilo de citação.

---

## 🤝 Contribuição

1. Fork este repositório
2. Crie uma branch: `git checkout -b feature/minha-nova-funcao`
3. Faça suas alterações e commit: `git commit -m "Adiciona minha nova feature"`
4. Envie para sua fork: `git push origin feature/minha-nova-funcao`
5. Abra um Pull Request aqui no GitHub

---

## 📝 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).


### Como usar

1. **Copie** o texto acima e crie um arquivo `README.md` na raiz do repositório.  
2. **Edite** os campos (descrição, URLs, variáveis de ambiente, caminhos de arquivo) conforme suas necessidades.  
3. **Faça commit** e **push** para o GitHub.  

Pronto! Agora qualquer pessoa que visitar seu repositório verá uma documentação clara de como instalar, usar e contribuir no projeto.
```

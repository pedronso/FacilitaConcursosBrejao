
# **📄 Sistema RAG - Consulta de Concursos Nacionais: FacilitaConcursosBrejao**  

Este projeto implementa um **sistema RAG** (Retrieval-Augmented Generation) para consulta de editais de concursos públicos. O sistema baixa os PDFs dos editais, extrai e divide o texto em chunks, cria embeddings para busca vetorial e utiliza **LLMs gratuitas da Groq** para gerar respostas.  

A interface web é construída com **Streamlit**, permitindo que os usuários façam perguntas e obtenham respostas diretamente da base de editais processada.

---

## **🚀 Tecnologias Utilizadas**
- 🧠 **[LangChain](https://python.langchain.com/)** → Pipeline RAG para recuperação e geração de texto  
- 🌎 **[Streamlit](https://streamlit.io/)** → Interface web  
- 🤖 **[Groq](https://groq.com/)** → LLM gratuita para geração de respostas  
- 📚 **[FAISS](https://faiss.ai/)** → Banco de dados vetorial para busca eficiente  
- 🔍 **[Hugging Face](https://huggingface.co/)** → Modelos de embeddings  
- 📄 **PDFMiner** → Extração de texto de PDFs  
- 🌐 **BeautifulSoup** → Web scraping para baixar editais  
- 🐍 **Python-dotenv** → Gerenciamento de variáveis de ambiente  

---

## **🛠️ Instalação e Configuração**
### **1️⃣ Clonar o Repositório**
```bash
git clone https://github.com/user/FacilitaConcursosBrejao.git
cd FacilitaConcursosBrejao
```

### **2️⃣ Criar um Ambiente Virtual**
```bash
python -m venv venv
```

Ativar o ambiente:  
**Windows (CMD ou PowerShell):**
```bash
venv\Scripts\activate
```
**Linux/macOS/WSL:**
```bash
source venv/bin/activate
```

### **3️⃣ Instalar as Dependências**
```bash
pip install -r requirements.txt
```

---

## **🔑 Configuração da API da Groq**
O sistema utiliza modelos da **Groq** para gerar respostas. Você precisa criar uma **chave de API** e armazená-la no arquivo `.env`.  

### **Passos:**
1. Acesse **[Groq Cloud](https://console.groq.com/)** e crie uma conta.  
2. Gere sua chave de API na aba **API Keys**.  
3. Crie um arquivo **`.env`** na raiz do projeto e adicione:
   ```
   GROQ_API_KEY=SUA_CHAVE_AQUI
   ```

---

## **📌 Como Rodar o Sistema**
### **1️⃣ Criar a Base Vetorial**
Execute o script para processar os PDFs e criar o banco de vetores:
```bash
python vectorstore.py
```

### **2️⃣ Iniciar a Interface Web**
```bash
streamlit run app.py
```
Isso abrirá o sistema no navegador!

---

## **🖥️ Como Usar o Sistema**
1️⃣ Digite sua pergunta no campo de texto.  
2️⃣ Clique em **"Consultar"**.  
3️⃣ O sistema buscará os chunks mais relevantes e gerará uma resposta.  
4️⃣ Expanda os **"Chunks Utilizados"** para ver quais trechos foram usados.  

---

## **📌 Estrutura do Projeto**
```
📂 FacilitaConcursosBrejao
 ├── 📜 app.py                 # Interface Web (Streamlit)
 ├── 📜 vectorstore.py         # Processamento e criação de embeddings
 ├── 📜 rag.py                 # Implementação do RAG (busca e geração de resposta)
 ├── 📜 scraper.py             # Web scraping para baixar editais
 ├── 📜 extractor.py           # Extração de texto e chunking dos PDFs
 ├── 📜 config.py              # Configuração de variáveis
 ├── 📜 requirements.txt       # Dependências do projeto
 ├── 📜 .env                   # API Key da Groq (não compartilhar)
 ├── 📁 pdfs/                  # PDFs baixados dos editais
 ├── 📁 data/                  # Base vetorial (FAISS)
 ├── 📁 chunks/                # Arquivos processados e divididos em chunks
 ├── 📁 logs/                  # Logs de execução
```

---

## **🛠️ Debug e Problemas Comuns**
- ❌ **Erro ao ativar o ambiente virtual no PowerShell?**  
  👉 Execute:  
  ```powershell
  Set-ExecutionPolicy Unrestricted -Scope Process
  ```
- ❌ **Streamlit não abre?**  
  👉 Certifique-se de que o ambiente virtual está ativado antes de rodar `streamlit run app.py`.  
- ❌ **Erro de API Key da Groq?**  
  👉 Verifique se criou corretamente o arquivo `.env` com `GROQ_API_KEY`.  

---

## **📢 Contribuição**
Brejão Corporations agradece o trabalho em equipe, buscando um mundo mais limpo e sustentável.

📧 **Alunos:** Pedro Caetano, Raylandson Cesar, João Rezende

🚀 **Divirtam-se explorando o mundo dos RAGs!** 🚀

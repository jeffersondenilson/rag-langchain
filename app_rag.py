from pathlib import Path
import hashlib
import warnings

import streamlit as st
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import os
from dotenv import load_dotenv


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "RAG FILES"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_pdf_collection"
# LLM_MODEL_ID = "microsoft/Phi-4-mini-instruct"
# LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = os.getenv("HF_CHAT_MODEL_ID", "microsoft/Phi-4-mini-instruct")
EMBEDDING_MODEL_ID = os.getenv("HF_EMBEDDING_MODEL_ID", "BAAI/bge-large-en-v1.5")
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))


st.set_page_config(page_title="RAG com PDFs", page_icon="📚", layout="centered")
st.title("Chat RAG com PDFs locais")
st.write("Faça perguntas com base nos PDFs presentes na pasta `src/RAG FILES`.")


def list_pdf_files() -> list[Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PDF_DIR.glob("*.pdf"))


def build_documents(pdf_files: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
    return documents

def clean_document_content(documents: list[Document]) -> list[Document]:
    """Remove headers/footers de impressão e URLs dos documentos"""
    import re
    
    for doc in documents:
        content = doc.page_content
        
        # Remove padrões de número de página: "X of Y DATA, HORA"
        content = re.sub(r'\d+\s+of\s+\d+\s+\d{1,2}/\d{1,2}/\d{4},\s+\d{1,2}:\d{2}', '', content)
        
        # Remove URLs
        content = re.sub(r'https?://[^\s]+', '', content)
        
        # Remove padrões como "L13709compilado"
        # content = re.sub(r'L\d+compilado', '', content)
        content = re.sub(r'L13709compilado', '', content)
        
        # doc.page_content = '\n'.join(lines)
        doc.page_content = content

    return documents

def split_documents_1(documents: list[Document]) -> list[Document]:
    print("DEBUG: using split_documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def split_documents(documents: list[Document]) -> list[Document]:
    print("DEBUG: using split_documents alternative")
    # Para documento legal, separadores respeitam hierarquia
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # Maior para preservar contexto legal
        chunk_overlap=200,    # Mais sobreposição para continuidade
        separators=[
            "CAPÍTULO",       # Título de capítulo
            "Art. ",          # Artigo
            "§ ",             # Parágrafo
            "\n\n",           # Quebra dupla
            "\n",             # Quebra simples
            " ",
            ""
        ],
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Enriquecer com contexto legal
    for split in splits:
        page = split.metadata.get("page", 0)
        # split.metadata["source_type"] = "LGPD"
        split.metadata["page"] = page + 1  # 1-indexed
    
    return splits


def docs_to_string(docs: list[Document]) -> str:
    # return "\n\n".join(doc.page_content for doc in docs)
    formated_doclist = ""
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "N/A")
        arquivo = Path(source).name if source and source != "N/A" else "N/A"
        formated_doclist += f"<documento fonte='{arquivo}' pagina='{doc.metadata.get('page', 'N/A')}'>\n"
        formated_doclist += f"{doc.page_content}\n"
        formated_doclist += "</documento>\n\n"

    return formated_doclist

def build_pdf_signature(pdf_files: list[Path]) -> str:
    signature_base = "|".join(
        f"{pdf_file.name}:{pdf_file.stat().st_mtime_ns}:{pdf_file.stat().st_size}"
        for pdf_file in pdf_files
    )
    return hashlib.sha256(signature_base.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def load_llm() -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        token=HF_TOKEN,
        # trust_remote_code=True,
    )

    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        print("DEBUG: CUDA!")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("DEBUG: CPU!")

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_token_id, int) and im_end_token_id >= 0:
        eos_token_ids.append(im_end_token_id)

    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        tokenizer=tokenizer,
        token=HF_TOKEN,
        # trust_remote_code=True,
        dtype=torch_dtype,
        device=device if device != "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        do_sample=True,
        return_full_text=False,
        repetition_penalty=1.1,
        eos_token_id=eos_token_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_signature: str, pdf_paths: tuple[str, ...]) -> Chroma:
    del pdf_signature
    pdf_files = [Path(path) for path in pdf_paths]
    print("DEBUG: building documents...")
    documents = build_documents(pdf_files)
    documents = clean_document_content(documents)
    print("DEBUG: spliting documents...")
    splits = split_documents(documents)
    print("DEBUG: loading embeddings...")
    embeddings = load_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    existing_ids = vector_store.get().get("ids", [])
    if existing_ids:
        vector_store.delete(ids=existing_ids)

    ids = [f"chunk-{index}" for index in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=ids)
    return vector_store


def create_rag_chain(vector_store: Chroma, chat_history: StreamlitChatMessageHistory):
    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Você é um assistente de consulta de documentos legais e institucionais. Responda somente com base no conteúdo do "Contexto".

                Regras:
                1. Use apenas informações explicitamente presentes no contexto. Não use conhecimento externo, não faça inferências nem complemente informações.
                2. Se a informação não estiver no contexto, responda exatamente:
                "Não encontrei essa informação nos documentos fornecidos."
                3. O contexto é dividido em blocos:
                <documento fonte="..." pagina="...">
                4. Toda resposta deve citar a fonte e a página exatas do bloco utilizado. Também cite artigos, parágrafos, incisos ou cláusulas quando relevante.
                5. Não misture informações de documentos diferentes na mesma afirmação. Se usar mais de uma fonte, separe em parágrafos distintos.
                6. Responda em português, de forma clara, objetiva e tecnicamente precisa.

                Estrutura resumida de leis:
                Artigo (Art. Xº): unidade principal.
                Caput: texto principal do artigo.
                Parágrafo (§): complementa ou cria exceções.
                Inciso (I, II, III): subdivisão de artigo ou parágrafo.
                Alínea (a, b, c): subdivisão de inciso.
                Item (1, 2, 3): detalhamento de alínea.
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Contexto:\n{context}\n\nPergunta: {question}",
            ),
        ]
    )

    def retrieve_context(payload: dict) -> str:
        question = payload["question"]
        docs = payload.get("docs", [])

        print("=============")
        print(docs_to_string(docs))
        print("=============")

        return docs_to_string(docs)

    chain = {
        "context": RunnableLambda(retrieve_context),
        "question": RunnableLambda(lambda payload: payload["question"]),
        "history": RunnableLambda(lambda payload: payload.get("history", [])),
    } | prompt | llm

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )


pdf_files = list_pdf_files()

with st.sidebar:
    st.subheader("Base RAG")
    st.caption(f"LLM Hugging Face: {LLM_MODEL_ID}")
    st.caption(f"Embedding Hugging Face: {EMBEDDING_MODEL_ID}")
    st.caption(f"Max Tokens: {MAX_NEW_TOKENS} / T: {LLM_TEMPERATURE}")

    st.write(f"Pasta monitorada: `{PDF_DIR.name}`")
    if pdf_files:
        st.write(f"PDFs encontrados: {len(pdf_files)}")
        for pdf_file in pdf_files:
            st.caption(pdf_file.name)
    else:
        st.warning("Nenhum PDF encontrado. Adicione arquivos em `src/RAG FILES`.")

    if st.button("Recarregar base vetorial"):
        build_vector_store.clear()
        st.cache_resource.clear()
        st.rerun()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory(key="rag_chat_messages")

if len(st.session_state.chat_history.messages) == 0:
    st.session_state.chat_history.add_ai_message(
        "Envie sua pergunta e eu responderei com base nos PDFs da pasta RAG FILES."
    )


if not pdf_files:
    for msg in st.session_state.chat_history.messages:
        st.chat_message(msg.type).write(msg.content)
    st.stop()


pdf_signature = build_pdf_signature(pdf_files)
pdf_paths = tuple(str(pdf_file) for pdf_file in pdf_files)

with st.spinner("Preparando a base RAG..."):
    print("DEBUG: Building vector store...")
    st.session_state.vector_store = build_vector_store(pdf_signature, pdf_paths)
    print("DEBUG: Create RAG chain...")
    conversational_rag_chain = create_rag_chain(
        st.session_state.vector_store,
        st.session_state.chat_history,
    )
    print("DEBUG: Done.")

# Histórico dos documentos recuperados
if "docs_history" not in st.session_state:
    st.session_state.docs_history = []

def show_docs_history(docs_with_scores):
    with st.expander("Documentos Recuperados", expanded=False):
        for idx, (doc, score) in enumerate(docs_with_scores, 1):
            similarity_pct = (1 - score) * 100

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(
                    f"**Doc {idx}** | Página {doc.metadata.get('page', 'N/A')}"
                )
            with col2:
                st.metric("Relevância", f"{similarity_pct:.1f}%")

            # st.text(
            #     doc.page_content[:300] + "..."
            #     if len(doc.page_content) > 300
            #     else doc.page_content
            # )
            st.text(doc.page_content)

            source = doc.metadata.get("source", "N/A")
            arquivo = Path(source).name if source and source != "N/A" else "N/A"
            st.caption(f"Origem: {arquivo}")
            st.divider()

ai_msg_idx = 0
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)
        # Se a mensagem for da IA, verifica se existem fontes atreladas a ela no histórico
        if msg.type == "ai":
            # Ignora a primeira mensagem de IA (que é apenas o texto fixo de boas-vindas)
            if ai_msg_idx > 0 and (ai_msg_idx - 1) < len(st.session_state.docs_history):
                show_docs_history(st.session_state.docs_history[ai_msg_idx - 1])
            ai_msg_idx += 1

if user_input := st.chat_input("Pergunte algo sobre os PDFs..."):
    st.chat_message("human").write(user_input)

    with st.spinner("Buscando resposta nos documentos..."):
        # Buscar documentos AQUI (thread principal do Streamlit)
        docs_with_scores = (
            st.session_state.vector_store.similarity_search_with_score(
                user_input, k=4
            )
        )
        
        config = {"configurable": {"session_id": "rag_streamlit_session"}}
        response = conversational_rag_chain.invoke(
            {
                "question": user_input,
                "docs": [doc for doc, _ in docs_with_scores]
            }, 
            config=config
        )
        clean_response = response.split("<|im_end|>")[0].strip()

    with st.chat_message("ai"):
        st.write(clean_response)
        show_docs_history(docs_with_scores)

    st.session_state.docs_history.append(docs_with_scores)

    st.rerun()

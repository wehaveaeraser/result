import os
import tempfile
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# 🔐 OpenAI API Key 설정
#os.environ["OPENAI_API_KEY"] = ""

from dotenv import load_dotenv
load_dotenv()

# Streamlit UI 구성
st.set_page_config(page_title="파일 업로드 + 헌법 Q&A 챗봇", layout="centered")
st.header("📄 업로드된 문서 기반 Q&A 챗봇 💬")

# GPT 모델 선택
selected_model = st.selectbox("사용할 GPT 모델을 선택하세요:", ("gpt-4o", "gpt-3.5-turbo-0125"))

# PDF 업로드
uploaded_file = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"])

# ✅ PDF 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    return loader.load_and_split()

# ✅ FAISS 임베딩 벡터 생성
@st.cache_resource
def create_vectorstore(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    for i, doc in enumerate(split_docs):
        doc.metadata["source"] = f"{doc.metadata.get('source', '업로드 파일')} (p.{doc.metadata.get('page', 'n/a')})"
    return FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-small"))

# ✅ RAG 체인 구성
def initialize_rag_chain(docs, selected_model):
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

    # 질문 정제용 시스템 프롬프트
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a new question, return a standalone version of the question."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    # QA 시스템 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say you don't know. 
Use polite Korean and include emoji.\n\n{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ✅ 대화 메시지 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "업로드한 문서에 대해 궁금한 것을 질문해 주세요 😊"}]

# ✅ 파일 업로드 후 실행
if uploaded_file:
    with st.spinner("PDF 분석 중..."):
        pages = load_and_split_pdf(uploaded_file)
        rag_chain = initialize_rag_chain(pages, selected_model)

        chat_history = StreamlitChatMessageHistory(key="chat_messages")
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

    # 이전 대화 출력
    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    # 사용자 질문 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"):
            with st.spinner("답변 생성 중..."):
                config = {"configurable": {"session_id": "upload_session"}}
                response = conversational_chain.invoke({"input": prompt}, config)
                answer = response["answer"]
                st.write(answer)

                # ✅ 참고 문서 출력
                with st.expander("🔍 참고한 문서 보기"):
                    for doc in response.get("context", []):
                        st.markdown(f"📄 {doc.metadata.get('source', '알 수 없음')}", help=doc.page_content)

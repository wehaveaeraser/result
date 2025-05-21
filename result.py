import os
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

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
load_dotenv()

# Streamlit UI 구성
st.set_page_config(page_title="파일 업로드 + 헌법 Q&A 챗봇", layout="centered") #st.set_page_config() :앱의 제목, 아이콘, 레이아웃, 초기 사이드바 상태 등을 설정하는 데 사용. 
                                                                             #맨 처음에 한 번만 사용해야 함.
st.header("📄 업로드된 문서 기반 Q&A 챗봇 💬")

# GPT 모델 선택
selected_model = st.selectbox("사용할 GPT 모델을 선택하세요:", ("gpt-4o", "gpt-3.5-turbo-0125"))

# PDF 업로드
uploaded_file = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"])

# 🔑 PDF 해시 생성 함수
def get_file_hash(file) -> str:   #업로드한 파일을 받음. ->str은 str타입의 값을 반환한다는 것을 의미함.(개발자의 의도를 전달하는 의도.)
    content = file.read()
    file.seek(0) #파일 포인터로 file.read()를 하였을 때, 파일 포인터가 끝으로 가게 되기때문에 이를 처음으로 돌려주기 위해서 seek을 사용.
                    #안 할경우 read()를 다시 하였을 때 빈 문자열 출력.
    return hashlib.md5(content).hexdigest()
    #파일 내용을 바탕으로 한 해시 문자열 생성
    #해시란, 데이터를 고정된 요약 값으로 변경하는 것을 이야기함->안정성,보안성(안전한 데이터 전송)
    #md5란, 128비트의 고정된 길이로 바꾸어준다. ->충돌위험이 있음,하지만 여기서는 안전을 위해서 쓴 것이 아니기 때문에 중복여부와 식별을 위해서 사용함.그래서 괜찮다.
# ✅ PDF 로드 및 분할

@st.cache_resource  #파일을 곧바로 pypdf_loader의 경로로 입력할 수 없음. -> streamlit은 파일 경로가 아닌 자체를 저장하기 때문에
#임시 파일을 만들어 pdf의 내용을 기록하고 이 파일의 경로를 변수에 저장하는 식의 우회적인 방식으로 접근해야 합니다.(p.239)
def load_and_split_pdf(file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    return loader.load_and_split()

# ✅ FAISS 저장/로드 통합 함수
@st.cache_resource  #캐시된 내용을 재사용
def load_or_create_vectorstore(_docs, file_hash):  #임베딩 벡터db가 있으면 불러오고 없으면 생성.
    index_path = os.path.join("faiss_index", file_hash) #저장된 벡터 인덱스가 위치할 경로.
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") #텍스트 임베딩 모델 로딩.

    if os.path.exists(index_path):   #벡터 인덱스가 이미 존재한다면면
        return FAISS.load_local(index_path, embedding_model) #저장된 벡터 인덱스 불러오기 재사용으로 인해 시간절약(faiss는 무조건 필요)

    # 없다면 새로 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #임베딩을 위해 쪼개준다.(겹치는 부분 없이)
    split_docs = text_splitter.split_documents(_docs) #위에서 정한 기준으로 분할 후 리스트 생성
    for doc in split_docs:
        doc.metadata["source"] = f"{doc.metadata.get('source', '업로드 파일')} (p.{doc.metadata.get('page', 'n/a')})" #문서의 출처 정보를 명확히 하기 위해 metadata["source"] 필드를 업데이트
    vectorstore = FAISS.from_documents(split_docs, embedding_model) #문서 조각들과 임베딩 모델을 사용하여 FAISS 벡터 DB를 생성(FAISS는 빠르고 효율적인 벡터 검색 라이브러리

    os.makedirs("faiss_index", exist_ok=True) #"faiss_index" 폴더가 없으면 생성(있으면 아무 일도 일어나지 않게 exist_ok = True)
    vectorstore.save_local(index_path) #위에서 만든 벡터 인덱스를 로컬 경로에 저장.(다음을 위해 캐시 파일로 유지)
    return vectorstore#벡터 저장소(faiss)불러오기

# ✅ RAG 체인 구성
def initialize_rag_chain(docs, file_hash, selected_model):
    vectorstore = load_or_create_vectorstore(docs, file_hash)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a new question, return a standalone version of the question."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

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
    file_hash = get_file_hash(uploaded_file) #pdf내용 읽어와서 md5해시값으로 계산.
    #파일식별키역할. 벡터 인덱스를 저장하거나 불러올 때 경로 이름으로 사용.
    with st.spinner("PDF 분석 중..."): #스피너ui로 사용자에게 로딩 메시지표시.
        pages = load_and_split_pdf(uploaded_file) #pdf처리함수,임시 파일 저장 및 페이지 텍스트변환.
        #langchain에서 사용할 수 있는 형태로 변환된 문서 리스트.
        rag_chain = initialize_rag_chain(pages, file_hash, selected_model)
        #벡터스토어 불러오거나 만들고 선택된 llm과 연결,문서 기반 질문응답 체인생성.
        chat_history = StreamlitChatMessageHistory(key="chat_messages")
        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )
        
    #사용자가 PDF 파일을 업로드하면,
    #파일의 해시값을 계산하고,
    #PDF를 분석하고 페이지별 텍스트로 나눕니다.
    #그 문서를 기반으로 RAG 체인을 초기화합니다.
    #대화 이력을 관리할 객체를 만들고,
    #그걸 이용해 대화형 질문응답 체인을 구성합니다.

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"):
            with st.spinner("답변 생성 중..."):
                config = {"configurable": {"session_id": "upload_session"}}
                response = conversational_chain.invoke({"input": prompt}, config)
                answer = response["answer"]
                st.write(answer)

                with st.expander("🔍 참고한 문서 보기"):
                    for doc in response.get("context", []):
                        st.markdown(f"📄 {doc.metadata.get('source', '알 수 없음')}", help=doc.page_content)

#LangChain의 conversational_chain을 통해 **PDF 기반 질문 응답(RAG)**을 실행하며,
#그에 따른 답변과 참고 문서를 Streamlit UI에 보여주는 부분
#과거 대화 내용(chat history)을 화면에 출력
#사용자가 질문 입력
#질문을 AI에게 전달하고, 스피너로 로딩 표시
#LangChain RAG 체인 실행 → 답변 생성
#답변 출력 + 참고한 문서 목록 제공 (출처 + 내용 미리보기)

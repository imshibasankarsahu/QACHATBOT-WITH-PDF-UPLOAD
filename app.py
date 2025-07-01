import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#set up stramlit app

st.title("QA CHAT HISTORY RAG WITH PDF UPLOAD ")
st.write("upload pdf and chat")

api_key=st.text_input("Enter your api key",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    session_id=st.text_input("session_id",value="chart1")

    if "store" not in st.session_state:
        st.session_state.store={}

    upload_file=st.file_uploader("upload a pdf file",type="pdf",accept_multiple_files=True)

    if upload_file:
        all_docs=[]
        for upload_file in upload_file:
            temppdf=f"./temp.pdf"

            with open(temppdf, "wb") as f:
                  f.write(upload_file.getvalue())
                  file_name=upload_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            all_docs.extend(docs)
        split_docs=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_docs=split_docs.split_documents(all_docs)
        db=Chroma.from_documents(final_docs,embeddings)
        rtvr=db.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
             )
        contextualize_q_prompt=ChatPromptTemplate.from_messages([

                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ])
        history_aware_retriever=create_history_aware_retriever(llm,rtvr,contextualize_q_prompt)

        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
       
       
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Enter your question:-")

        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={"configurable":{"session_id":session_id}},
            )
            st.success(response["answer"])
            st.write("chat History",session_history.messages)



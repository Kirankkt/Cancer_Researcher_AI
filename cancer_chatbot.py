
import requests
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd

# Load data
df = pd.read_csv('Final_Stacked_Data_without_Duplicates.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# Strip df columns of spaces
df['Title'] = df['Title'].str.strip()
df['Authors'] = df['Authors'].str.strip()
df['Abstract'] = df['Abstract'].str.strip()
df['Link'] = df['Link'].str.strip()

# Set the page title and introduction
st.set_page_config(page_title="Cancer Research Assistant")
st.title("Cancer Research Assistant")
st.write("Ask questions about cancer and retrieve research papers relevant to the specific type of cancer.")

# Combine relevant columns into a single row
df['combined'] = df.apply(
    lambda row: f"Title: {row['Title']}\n"
                f"Authors: {row['Authors']}\n"
                f"Abstract: {row['Abstract']}\n"
                f"Link: {row['Link']}\n",
    axis=1
).str.strip()

# Step 2: Set Up OpenAI API Key
api_key = st.text_input("Please enter your OpenAI API key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Set up the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create Embeddings and Vector Store
    embeddings = OpenAIEmbeddings()
    documents = df['combined'].tolist()
    vector_store = FAISS.from_texts(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        '''You are a cancer research assistant. When the user asks about a specific type of cancer (e.g., brain tumor), you should:
        1. First explain the concept or type of cancer briefly.
        2. Provide at least 3 relevant links to papers related to that cancer from the available dataset. If not, provide however many there are.

        Question: {input}
        Context: {context}

        Output:
        1. Brief Explanation:
        2. Relevant Research Papers (with links):

        If the user asks for summaries or explanations of papers that you provided, you should look at the list of papers you provided for the previous question and summarize what is required.
        If the user's question is not related to cancer, do not try to find similarities, just say, "I don't know."
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Step 5: Interactive Loop
    st.subheader("Ask a question:")

    # Maintain session state for conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User input field for the query
    user_input = st.text_input("Ask a cancer-related question or type 'exit' to cancel:", key="user_input")

    # Only process if the user input is not 'exit'
    if user_input and user_input.lower() != 'exit':
        session_id = "user_session"  # You can customize this if needed
        
        # Invoke the conversational RAG chain
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        # Store user input and response in session history
        st.session_state.history.append({"user": user_input, "response": response})

        # Display the conversation history
        st.write("### Conversation History:")
        for entry in st.session_state.history:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Assistant:** {entry['response']}")

    elif user_input.lower() == 'exit':
        st.write("Exiting the conversation.")
else:
    st.warning("Please enter your OpenAI API key to proceed.")

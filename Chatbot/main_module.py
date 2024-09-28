
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

df = pd.read_csv('Final_Stacked_Data_without_Duplicates.csv')
df = df.drop(['Unnamed: 0'], axis=1)



# Combine relevant columns of df 'Title', 'Authors', 'Published', 'Journal', 'Abstract', 'Link' into a single row
df['combined'] = df.apply(
    lambda row: f"Title: {row['Title']}\n"
                f"Authors: {row['Authors']}\n"
                f"Abstract: {row['Abstract']}\n"
                f"Link: {row['Link']}\n",
    axis=1
)

# Strip spaces
df['combined'] = df['combined'].str.strip()

# Check the combined data
#print("Combined Data Example:\n", df['combined'].head())


# Ask for the OpenAI API key if not already set
if "OPENAI_API_KEY" not in os.environ:
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Convert the combined text into a list
documents = df['combined'].tolist()

# Create FAISS vector store with the embeddings
vector_store = FAISS.from_texts(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Contextualize question
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

# Answer question
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

# Statefully manage chat history
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

# Interactive Loop
def interactive_question():
    session_id = "user_session"

    while True:
        user_input = input("\nAsk a cancer-related question (or type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the interactive question loop.")
            break

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        print("\nResponse:")
        print(response)


interactive_question()
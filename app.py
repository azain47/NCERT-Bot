import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding,OpenAIEmbeddingModelType
from llama_index.llms import openai
from llama_index.core import Settings
from llama_index.core import PromptTemplate

from openai import OpenAI

Settings.llm = openai.OpenAI(model = "gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model = OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL)

chat_model = OpenAI()
messages = []

@st.cache_resource()
def load_vector_indices(directory = None, store_dir = 'storage', process=False):
    if process and directory:
        english_docs = SimpleDirectoryReader(directory).load_data(show_progress=True)

        index = VectorStoreIndex.from_documents(english_docs)
        index.storage_context.persist(f'./{directory}/{store_dir}')
        
    else :
        storage_context = StorageContext.from_defaults(persist_dir=f'./{directory}/{store_dir}')
        index = load_index_from_storage(storage_context)
    
    return index

def add_message(content, role = "user", chat_history = False):
    # if role == 'assistant':
    #     st.session_state['chat_history'].append(
    #         {"role":role,"content":content}
    #     )
    #     messages.append({"role":role,"content":content})
    
    # elif chat_history:
    #     st.session_state['chat_history'].append(
    #         {"role":role,"content":content}
    #     )
    
    # else:
        messages.append({"role":role,"content":content})

def write_messages(container, role, content):
    container.chat_message(role).write(content)
         
def main():
    st.set_page_config("NCERT Bot")
    st.title("NCERT Chatbot")

    # For keeping track of conversation in message container
    # if "chat_history" not in st.session_state:
    #     st.session_state["chat_history"] = []
    
    retreiver = None
    selected_book = st.selectbox("Books:", options = ["English"], index = None)
    if selected_book:
        st.session_state['book_selected'] = selected_book
        vector_index = load_vector_indices(directory=os.path.join(str(selected_book)))
        retreiver = vector_index.as_retriever()
    
    template_prompt = """You are a helpful AI assistant to only answer questions from Academic NCERT Books. Use the following pieces of context to answer the question at the end, also give the source.
    Very Important: If the question is about writing code use backticks (```) at the front and end of the code snippet and include the language use after the first ticks.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Use as much detail when as possible when responding.

    Context: {context}

    Context Source: {sources}

    Question: {question}
    """

    if retreiver:
        message_container = st.container(height=400)
        # user_msgs = message_container.chat_message("user")
        # ai_responses = message_container.chat_message("assistant")

        query = st.chat_input("Ask any question:", max_chars=300)

        if query:

            write_messages(message_container, "user", query)

            retrieved_context = retreiver.retrieve(query)
            
            sources = f"Page number : {retrieved_context[0].node.metadata.get('page_label')} , File Name: {retrieved_context[0].node.metadata.get('file_name')}"
            
            user_content = PromptTemplate(template_prompt).format(context=retrieved_context[0].text, 
                    question=query, sources = sources)
            
            add_message(user_content)
            
            ai_response = chat_model.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=messages
            ).choices[0].message.content
                     
            add_message(ai_response, role="assistant")

            write_messages(message_container, "assistant", ai_response)

if __name__ == "__main__":
    main()
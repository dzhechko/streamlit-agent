from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatYandexGPT

import streamlit as st

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

"""
Простой пример использования StreamlitChatMessageHistory, чтобы помочь LLMChain запоминать сообщения в диалоге. 
Пример адаптирован для использования с YandexGPT
Сообщения сохраняются в состоянии сессии при автоматическом повторном запуске. Вы можете просмотреть содержимое состояния сессии
ниже. [Исходный код приложения ](https://github.com/dzhechko/streamlit-agent/edit/main/streamlit_agent/basic_memory.py).
"""

# Настраиваем алгоритмы работы памяти
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Как я могу вам помочь?")

view_messages = st.expander("Просмотр сообщения в состоянии сессии")

# folder_id = st.secrets["yagpt_folder_id"]
# api_id = st.secrets["yagpt_api_id"]
# api_key = st.secrets["yagpt_api_key"]

# Получение folder id
if "YC_FOLDER_ID" in st.secrets:
    yagpt_folder_id = st.secrets.YC_FOLDER_ID
else:
    yagpt_folder_id = st.sidebar.text_input("YC folder ID", type="password")
if not yagpt_folder_id:
    st.info("Введите YaGPT folder ID для продолжения")
    st.stop()

# Получение ключа YaGPT API
if "YC_API_KEY" in st.secrets:
    yagpt_api_key = st.secrets.YC_API_KEY
else:
    yagpt_api_key = st.sidebar.text_input("YaGPT API Key", type="password")
if not yagpt_api_key:
    st.info("Введите YaGPT API ключ для продолжения")
    st.stop()
    
# st.info(yagpt_api_key)
# st.info(yagpt_folder_id)

# Настраиваем LangChain, передавая Message History
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты очень полезный чат-бот. При ответе на вопросы будет краток, используй 30 слов или меньше."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"
# model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
model = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = 0.6)

chain = prompt | model
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Отображать текущие сообщения из StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Примечание: новые сообщения автоматически сохраняются в историю по длинной цепочке во время запуска
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

# Отобразить сообщения в конце, чтобы вновь сгенерированные отображались сразу
with view_messages:
    """
    История сообщений, инициализированная с помощью:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Содержание `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)

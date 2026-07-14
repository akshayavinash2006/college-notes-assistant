from config import llm
from vectorstore import retriever

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------------------
# Prompt to rewrite follow-up questions
# -----------------------------------------

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the chat history and the latest user question,
rewrite it into a standalone question.

Do NOT answer the question.

Only rewrite it if necessary.
            """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt,
)

# -----------------------------------------
# Prompt for answering
# -----------------------------------------

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful college notes assistant.

Use ONLY the provided context to answer.

If the answer isn't found in the context, reply:

"I couldn't find that in your notes."

Context:
{context}
            """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

# -----------------------------------------
# Chat History
# -----------------------------------------

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# -----------------------------------------
# Chat Loop
# -----------------------------------------

print("College Notes Assistant")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    response = conversational_rag_chain.invoke(
        {
            "input": question,
        },
        config={
            "configurable": {
                "session_id": "default",
            }
        },
    )

    print("\nAssistant:")
    print(response["answer"])
    print()
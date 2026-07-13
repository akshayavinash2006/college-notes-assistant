from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from config import llm

chat_history = [
    SystemMessage(
        content="You are a helpful college notes assistant."
    )
]

parser = StrOutputParser()

while True:

    question = input("You: ")

    if question == "exit":
        break

    chat_history.append(
        HumanMessage(content=question)
    )

    response = llm.invoke(chat_history)

    answer = parser.invoke(response)

    print(answer)

    chat_history.append(response)
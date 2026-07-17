from config import llm

from tools import (
    generate_flashcards,
    explain_simply,
)

from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[
        generate_flashcards,

        explain_simply,
    ],
    system_prompt="""
You are a college notes assistant.

You have access to several tools.

Choose the most appropriate tool based on the user's request.

Do not make up information.
Use the tools whenever possible.
"""
)
print("College Notes Agent")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\nAssistant:\n")

    message = response["messages"][-1]

    if isinstance(message.content, list):
        for block in message.content:
            if block["type"] == "text":
                print(block["text"])
    else:
        print(message.content)

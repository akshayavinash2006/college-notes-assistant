from config import llm
from vectorstore import retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

flashcard_prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant.

Using ONLY the notes below, generate flashcards.

Each flashcard should follow this format:

Q: ...
A: ...

Generate at least 10 flashcards.

Context:
{context}

Topic:
{topic}
""")

parser = StrOutputParser()

flashcard_chain = flashcard_prompt | llm | parser

def retrieve_context(topic: str):
    docs = retriever.invoke(topic)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

@tool
def generate_flashcards(topic: str) -> str:
    """
    Generate flashcards for a topic using the student's notes.
    """
    print("Flashcard tool called!")

    docs = retriever.invoke(topic)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    result=flashcard_chain.invoke(
        {
            "context": context,
            "topic": topic,
        }
    )
    print("Flashcard tool result:")
    print(result)
    return result

@tool
def explain_simply(topic: str) -> str:
    """
    Explain a topic from the student's notes in very simple language.
    Useful when the user asks for an easy explanation or says they don't understand a concept.
    """

    context = retrieve_context(topic)

    prompt = ChatPromptTemplate.from_template(
        """
            You are an expert teacher.

            Use ONLY the context below.

            Explain the topic as if you're teaching a first-year college student.

            Rules:
            - Use simple language.
            - Avoid jargon whenever possible.
            - If you must use technical terms, explain them.
            - Use analogies and examples.
            - Keep the explanation easy to follow.
            - If the answer is not found in the notes, say:
            "I couldn't find that in your notes."

            Context:
            {context}

            Topic:
            {topic}
            """
    )

    chain = prompt | llm | StrOutputParser()

    result=chain.invoke(
        {
            "context": context,
            "topic": topic,
        }
    )
    print("Explain simply tool result:")
    print(result)
    return result


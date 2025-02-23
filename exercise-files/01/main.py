from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from colorama import Fore
from langchain_core.output_parsers import StrOutputParser

# https://python.langchain.com/docs/introduction/
load_dotenv()
llm = OpenAI()


# prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

# def generate(text):
#     """Provide text input to the model and return the generated text using promt templates.
#     https://python.langchain.com/docs/how_to/#prompt-templates"""
#     prompt = prompt_template.format(topic=text)
#     print(prompt)
#     return llm.invoke(prompt)    


prompt_template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
output_parser = StrOutputParser()

def generate(text):
    """declarative way to compose chains together using LCEL: using ChatPromptTemplate.
    https://python.langchain.com/docs/concepts/lcel/
    https://python.langchain.com/docs/concepts/output_parsers/"""

    chain = prompt_template | llm | output_parser
    print(prompt_template)
    return chain.invoke({"topic": text})     


def start():
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:
            response = generate(user_input)
            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Anthony Edward Stark is a fictional character portrayed by Robert Downey Jr. in the Marvel Cinematic Universe (MCU) film franchise , based on the Marvel Comics character of the same name and commonly known by his alias , Iron Man . In the films, Stark describes himself as a "genius, billionaire, playboy, and philanthropist" and is the CEO of Stark Industries . Early in the franchise, he is the primary weapons manufacturer for the United States Armed Forces , until he changes his mind and redirects his technical expertise to creating mechanized armor that he uses to defend himself against those who threaten world peace. He becomes a founding member and leader of the Avengers .

    Stark is one of the central figures in the MCU, having appeared in nine films and the Disney+ animated series What If...? The character and Downey's performance have been credited with helping shape the MCU into a billion-dollar franchise, with Stark's evolution often considered the defining arc of the series.
    """

    summary_template = """
    Given the information below about a person, I want you to produce:
    1. A short summary
    2. Two interesting facts about them

    Use ONLY the information provided. Do not invent facts.

    Information: {information}
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})

    print(response.content)

if __name__ == "__main__":
    main()

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")
    information = """
        Kawhi Anthony Leonard (/kəˈhwaɪ/ kə-WHY;[1] born June 29, 1991) is an American professional basketball player for the Los Angeles Clippers of the National Basketball Association (NBA). A two-time NBA champion, he is a six-time All-Star and a six-time member of the All-NBA Team (including three First Team selections). Nicknamed the "Claw" or "Klaw" for his ball-hawking skills and exceptionally large hands, Leonard is often regarded as one of the greatest[under discussion] two-way players in NBA history,[2][3][4] earning seven All-Defensive Team selections and winning Defensive Player of the Year honors in 2015 and 2016. In 2021, he was named to the NBA 75th Anniversary Team.

Leonard played two seasons of college basketball for the San Diego State Aztecs and was named a consensus second-team All-American as a sophomore. He opted to forgo his final two seasons of college eligibility to enter the 2011 NBA draft. He was selected by the Indiana Pacers with the 15th overall pick before being traded to the San Antonio Spurs on draft night.

With the Spurs, Leonard won an NBA championship in 2014, when he was named the Finals Most Valuable Player. After seven seasons with the Spurs, Leonard was traded to the Toronto Raptors in 2018. In 2019, he led the Raptors to their first NBA championship and won his second Finals MVP award (one of only three players to win Finals MVP with multiple teams, along with Kareem Abdul-Jabbar and LeBron James). He subsequently moved to his hometown of Los Angeles and signed with the Clippers as a free agent in July 2019. 

    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm
    res = chain.invoke(input={"information": information})

    print(res)

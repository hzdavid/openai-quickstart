from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def main1():
    llm = OpenAI(model_name="gpt-3.5-turbo")
    print(llm("Tell me a Joke in Chinese"))

def main2():
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
    messages = [SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Who won the world series in 2020?"),
                AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
                HumanMessage(content="Where was it played?")]
    chat_result = chat_model(messages)
    type(chat_result)
    print(chat_result)

if __name__ == "__main__":
    main1()
    main2()
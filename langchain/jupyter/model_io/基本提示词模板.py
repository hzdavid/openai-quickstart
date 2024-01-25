from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


def main1():
    prompt_template = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}."
    )
    # 使用 format 生成提示
    prompt = prompt_template.format(adjective="funny", content="chickens")
    print(prompt)
    print(prompt_template)

def main2():
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
    prompt_template = PromptTemplate.from_template(
        "讲{num}个给程序员听得笑话"
    )
    prompt = prompt_template.format(num=2)
    print(f"prompt: {prompt}")

    result = llm(prompt)
    print(f"result: {result}")

def main3():
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ])

    # 生成提示
    messages = template.format_messages(
        name="Bob",
        user_input="What is your name?"
    )
    print(messages)
    print(messages[0].content)
    #最后一个元素
    print(messages[-1].content)
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
    result=chat_model(messages)
    print(f"result: {result}")

if __name__ == "__main__":
    # main1()
    # main2()
    main3()
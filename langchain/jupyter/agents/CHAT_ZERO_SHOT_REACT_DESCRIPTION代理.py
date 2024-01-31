from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents import initialize_agent, AgentType, Tool, load_tools
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.openai import OpenAI

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "79a206f83aec7fe46402cc5f712409e255ae6c5fd98075fbc08659c406119b66"

def main1():
    llm = OpenAI(temperature=0)
    # 加载 LangChain 内置的 Tools
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # 实例化 ZERO_SHOT_REACT Agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("谁是莱昂纳多·迪卡普里奥的女朋友？她现在年龄的0.43次方是多少?")

def main2():
    chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    # 加载 LangChain 内置的 Tools
    #这里的serpapi和llm-math是LangChain库提供的内置工具，每个工具都有特定的功能：
    tools = load_tools(["serpapi", "llm-math"], llm=chat_model)
    agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("谁是莱昂纳多·迪卡普里奥的女朋友？她现在年龄的0.43次方是多少?")
    return



if __name__ == "__main__":
    #main1()
    main2()
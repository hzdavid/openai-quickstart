from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool
import os

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "79a206f83aec7fe46402cc5f712409e255ae6c5fd98075fbc08659c406119b66"

def main1():
    #llm = OpenAI(model_name="gpt-3.5-turbo")
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    # 实例化查询工具
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    # 实例化 SELF_ASK_WITH_SEARCH Agent
    self_ask_with_search = initialize_agent(
        tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
    )
    # 实际运行 Agent，查询问题（正确）
    self_ask_with_search.run(
        "成都举办的大运会是第几届大运会？"
    )
    print(11111111111111111111)
    self_ask_with_search.run(
        "2023年大运会举办地在哪里？"
    )

def main2():
    chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    # 实例化查询工具
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    self_ask_with_search_chat = initialize_agent(
        tools, chat_model, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
    )
    # GPT-4 based ReAct 答案（正确）
    self_ask_with_search_chat.run(
        "2023年大运会举办地在哪里？"
    )
    return



if __name__ == "__main__":
    main1()
    #main2()
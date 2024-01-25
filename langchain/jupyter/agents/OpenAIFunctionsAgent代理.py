from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents import initialize_agent, AgentType, Tool, load_tools, OpenAIFunctionsAgent, AgentExecutor
import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.openai import OpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "79a206f83aec7fe46402cc5f712409e255ae6c5fd98075fbc08659c406119b66"

def main1():
    # 使用 GPT-3.5-turbo
    llm = ChatOpenAI(temperature=0)
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    tools = [get_word_length]
    system_message = SystemMessage(content="你是非常强大的AI助手，但在计算单词长度方面不擅长。")
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt,verbose=True)
    # 实例化 OpenAIFunctionsAgent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.run("单词“educa”中有多少个字母?")


def main2():
    MEMORY_KEY = "chat_history"
    system_message = SystemMessage(content="你是非常强大的AI助手，但在计算单词长度方面不擅长。")
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
    )
    memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    tools = [get_word_length]
    llm = ChatOpenAI(temperature=0)
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    #AgentExecutor也是一个chain，它也有记忆能力
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    agent_executor.run("单词“educa”中有多少个字母?")
    agent_executor.run("那是一个真实的单词吗？")
    return



if __name__ == "__main__":
    #main1()
    main2()
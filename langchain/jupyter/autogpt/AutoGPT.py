import os

import faiss
from langchain_community.chat_models import ChatOpenAI
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.tools.file_management import WriteFileTool, ReadFileTool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.tools import Tool
from langchain_experimental.autonomous_agents import AutoGPT

# 更换为自己的 Serp API KEY
os.environ["SERPAPI_API_KEY"] = "79a206f83aec7fe46402cc5f712409e255ae6c5fd98075fbc08659c406119b66"

def main1():
    # 构造 AutoGPT 的工具集
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    # OpenAI Embedding 模型
    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    #实例化自主智能体
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0, verbose=True),
        memory=vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}),  # 实例化 Faiss 的 VectorStoreRetriever
    )
    # 打印 Auto-GPT 内部的 chain 日志
    agent.chain.verbose = True
    result=agent.run(["2023年成都大运会，中国金牌数是多少"])
    print(result)


def main2():
    return


if __name__ == "__main__":
    main1()
    #main2()


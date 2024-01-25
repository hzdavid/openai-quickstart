from langchain import text_splitter
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate

# Embeddings类是一个专门用于与文本嵌入模型进行交互的类。有许多嵌入模型提供者（OpenAI、Cohere、Hugging Face等）-这个类旨在为所有这些提供者提供一个标准接口。
#
# 嵌入将一段文本创建成向量表示。这非常有用，因为它意味着我们可以在向量空间中思考文本，并且可以执行语义搜索等操作，在向量空间中寻找最相似的文本片段。
#
# LangChain中基础的Embeddings类公开了两种方法：一种用于对文档进行嵌入，另一种用于对查询进行嵌入。前者输入多个文本，而后者输入单个文本。之所以将它们作为两个独立的方法，是因为某些嵌入提供者针对要搜索的文件和查询（搜索查询本身）具有不同的嵌入方法。
def main1():
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )
    #5个向量
    print(len(embeddings))
    #每个向量是1536维
    print(len(embeddings[0]))
    embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    # embedded_query也是1536维的向量， 转成向量的目的：嵌入将一段文本创建成向量表示。这非常有用，因为它意味着我们可以在向量空间中思考文本，并且可以执行语义搜索等操作，在向量空间中寻找最相似的文本片段。
    print(len(embedded_query))

def main2():
    return



if __name__ == "__main__":
    main1()
    #main2()
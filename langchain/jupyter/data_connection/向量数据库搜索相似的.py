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


def main1():
    # 加载长文本
    raw_documents = TextLoader('../tests/state_of_the_union.txt').load()
    # 实例化文本分割器
    text_splitter=CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    # 分割文本
    documents = text_splitter.split_documents(raw_documents)
    embeddings_model = OpenAIEmbeddings()
    # 将分割后的文本，使用 OpenAI 嵌入模型获取嵌入向量，并存储在 Chroma 中
    db = Chroma.from_documents(documents, embeddings_model)
    #使用文本进行语义相似度搜索
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)
    print(docs[0].page_content)
    #使用嵌入向量进行语义相似度搜索
    embedding_vector = embeddings_model.embed_query(query)
    #与通过similarity_search 来搜索并没有什么不同。
    docs = db.similarity_search_by_vector(embedding_vector)
    print(docs[0].page_content)




def main2():
    return



if __name__ == "__main__":
    main1()
    #main2()
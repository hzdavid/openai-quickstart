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


def main1():
    docs = TextLoader('../tests/state_of_the_union.txt').load()
    type(docs[0])
    print(docs[0].page_content[:100])



def main2():
    html_text = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>🦜️🔗 LangChain</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                h1 {
                    color: darkblue;
                }
            </style>
        </head>
        <body>
            <div>
                <h1>🦜️🔗 LangChain</h1>
                <p>⚡ Building applications with LLMs through composability ⚡</p>
            </div>
            <div>
                As an open source project in a rapidly developing field, we are extremely open to contributions.
            </div>
        </body>
    </html>
    """
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=60, chunk_overlap=0
    )
    html_docs = html_splitter.create_documents([html_text])
    print(len(html_docs))
    print(html_docs)



if __name__ == "__main__":
    #main1()
    main2()
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

#使用 FewShotPromptTemplate 类生成 Few-shot Prompt
def main1():
    # 创建一个输出解析器，用于处理带逗号分隔的列表输出
    output_parser = CommaSeparatedListOutputParser()

    # 获取格式化指令，该指令告诉模型如何格式化其输出
    format_instructions = output_parser.get_format_instructions()

    # 创建一个提示模板，它会基于给定的模板和变量来生成提示
    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",  # 模板内容
        input_variables=["subject"],  # 输入变量
        partial_variables={"format_instructions": format_instructions}  # 预定义的变量，这里我们传入格式化指令
    )
    _input = prompt.format(subject="ice cream flavors")
    llm = OpenAI(temperature=0)
    #_input 的实际内容是：'List five ice cream flavors.
#Your response should be a list of comma separated values, eg: `foo, bar, baz`'
    # 有些是CommaSeparatedListOutputParser 产生的。 可以看出，本质上还是在提示词上做文章，告诉llm 输出结果。
    output = llm(_input)
    print(output)
    output_parser.parse(output)
    return


def main2():
    output_parser = DatetimeOutputParser()
    template = """Answer the users question:

    {question}

    {format_instructions}"""

    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    print(prompt.format(question="around when was bitcoin founded?"))
    #prompt 实际内容 Answer the users question:
#     around when was bitcoin founded?
#
#     Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.
#
# Examples: 0863-10-20T14:57:05.922363Z, 1321-11-15T15:16:40.898262Z, 0420-05-16T02:54:40.609119Z
#
# Return ONLY this string, no other words!
    # 可以看出，原理和CommaSeparatedListOutputParser一样，也是DatetimeOutputParser也在提示词中加在一些提示性的文案，让llm 知道要以时间格式回复。
    chain = LLMChain(prompt=prompt, llm=OpenAI())
    output = chain.run("around when was bitcoin founded?")
    print(output_parser.parse(output))
    return


def main3():
    return




if __name__ == "__main__":
    #main1()
    main2()
    #main3()
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, ConversationChain, MultiPromptChain, \
    TransformChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate


def main1():
    with open("../tests/the_old_man_and_the_sea.txt") as f:
        novel_text = f.read()
    print(novel_text)
    len(novel_text)

    # 定义一个转换函数，输入是一个字典，输出也是一个字典。
    def transform_func(inputs: dict) -> dict:
        # 从输入字典中获取"text"键对应的文本。
        text = inputs["text"]
        # 使用split方法将文本按照"\n\n"分隔为多个段落，并只取前三个，然后再使用"\n\n"将其连接起来。
        shortened_text = "\n\n".join(text.split("\n\n")[:3])
        # 返回裁剪后的文本，用"output_text"作为键。
        return {"output_text": shortened_text}

    # 使用上述转换函数创建一个TransformChain对象。
    # 定义输入变量为["text"]，输出变量为["output_text"]，并指定转换函数为transform_func。
    transform_chain = TransformChain(
        input_variables=["text"], output_variables=["output_text"], transform=transform_func
    )
    transformed_novel = transform_chain(novel_text)
    print(transformed_novel['text'])
    type(transformed_novel)
    print(transformed_novel["output_text"])

    template = """总结下面文本:

    {output_text}

    总结:"""
    prompt = PromptTemplate(input_variables=["output_text"], template=template)
    llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, verbose=True)
    llm_chain(transformed_novel['output_text'][:1000])
    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
    sequential_chain.run(novel_text[:100])







def main2():
    return


if __name__ == "__main__":
    main1()
    #main2()
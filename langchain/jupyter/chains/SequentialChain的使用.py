from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate


def main1():
    llm = OpenAI(temperature=0.7, max_tokens=1000)

    template = """你是一位剧作家。根据戏剧的标题，你的任务是为该标题写一个简介。

    标题：{title}
    剧作家：以下是对上述戏剧的简介："""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    # 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。
    # llm = OpenAI(temperature=0.7, max_tokens=1000)
    template = """你是《纽约时报》的戏剧评论家。根据剧情简介，你的工作是为该剧撰写一篇评论。

    剧情简介：
    {synopsis}

    以下是来自《纽约时报》戏剧评论家对上述剧目的评论："""

    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    #SimpleSequentialChain就是把2个chain串起来， 前一个chain的输出，是下一个chain的输入， 输出与输入的变量，也可以人工指定，见main2
    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
    result = overall_chain.run("三体人不是无法战胜的")
    print(f"result: {result}")

def main2():
    # # 这是一个 LLMChain，根据剧名和设定的时代来撰写剧情简介。
    llm = OpenAI(temperature=.7, max_tokens=1000)
    template = """你是一位剧作家。根据戏剧的标题和设定的时代，你的任务是为该标题写一个简介。

    标题：{title}
    时代：{era}
    剧作家：以下是对上述戏剧的简介："""

    prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
    # output_key  就是模型的输出
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis", verbose=True)

    # 这是一个LLMChain，用于根据剧情简介撰写一篇戏剧评论。

    template = """你是《纽约时报》的戏剧评论家。根据该剧的剧情简介，你需要撰写一篇关于该剧的评论。

    剧情简介：
    {synopsis}

    来自《纽约时报》戏剧评论家对上述剧目的评价："""

    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", verbose=True)

    m_overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True)
    #指定不同变量的名字
    result = m_overall_chain({"title": "三体人不是无法战胜的", "era": "二十一世纪的新中国"})
    print(1111111111)
    print(result)



if __name__ == "__main__":
    main1()
    #main2()
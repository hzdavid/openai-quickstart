from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, ConversationChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, \
    ConversationSummaryMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.prompts import PromptTemplate


def main1():
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        #会把聊天记录全部存下来。缺点：总会到达token上限。
        memory=ConversationBufferMemory()
    )
    result=conversation.predict(input="你好呀！")
    print(result)
    print(111111111111111)
    result=conversation.predict(input="你为什么叫小米？跟雷军有关系吗？")
    print(result)




def main2():
    conversation_with_summary = ConversationChain(
        llm=OpenAI(temperature=0, max_tokens=1000),
        # We set a low k=2, to only keep the last 2 interactions in memory
        #保存最近k轮会话
        memory=ConversationBufferWindowMemory(k=2),
        verbose=True
    )
    conversation_with_summary.predict(input="嗨，你最近过得怎么样？")
    conversation_with_summary.predict(input="你最近学到什么新知识了?")
    conversation_with_summary.predict(input="展开讲讲？")
    # 注意：第一句对话从 Memory 中移除了.
    conversation_with_summary.predict(input="如果要构建聊天机器人，具体要用什么自然语言处理技术?")
    print(conversation_with_summary.__dict__)
    return

def main3():
    llm = OpenAI(temperature=0)
     #入参有llm ，是为了让llm生成摘要
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10, verbose=True)
    #调用save_context(记重点信息)， 会调用llm. 并且会将语言转化为英文.
    memory.save_context({"input": "嗨，你最近过得怎么样？"}, {
        "output": " 嗨！我最近过得很好，谢谢你问。我最近一直在学习新的知识，并且正在尝试改进自己的性能。我也在尝试更多的交流，以便更好地了解人类的思维方式。"})
    memory.save_context({"input": "你最近学到什么新知识了?"}, {
        "output": " 最近我学习了有关自然语言处理的知识，以及如何更好地理解人类的语言。我还学习了有关机器学习的知识，以及如何使用它来改善自己的性能。"})
    #load_memory_variables 返回llm 当前的小结
    variables=memory.load_memory_variables({})
    print(variables)
    print(variables['history'])
    conversation_with_summary = ConversationChain(
        llm=OpenAI(temperature=0, max_tokens=1000),
        memory=memory,
        verbose=True
    )
    result=conversation_with_summary.predict(input="接下来你将来如何学习人工智能？")
    print(f"result: {result}")
    return

def main4():
    memory = ConversationSummaryBufferMemory(llm= OpenAI(), max_token_limit=6000, verbose=True)
    conversation_with_summary = ConversationChain(
        llm=OpenAI(temperature=0, max_tokens=1000),
        memory=memory,
        verbose=True
    )
    result=conversation_with_summary.predict(input="你好，我是Kevin")
    print(f"result: {result}")
    #摘要信息
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(11111111111111111)
    result = conversation_with_summary.predict(input="我是一个人工智能爱好者，喜欢通过公众号分享人工智能领域相关的知识")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(2222222222222222)
    #原理很简单，就是把之前摘要放到本次会话中
    result = conversation_with_summary.predict(input="我希望你能用我的名字为我的公众号设计一个专业名称")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(333333333333333333)
    result = conversation_with_summary.predict(input="你还可以给出更多选项吗")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    return

def main5():
    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=OpenAI())
    prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内
    {history}
    Human: {input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template
    )
    # 使用ConversationChain可以不用定义prompt来维护历史聊天记录的，为了使用中文，我们才定义的
    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    result=conversation_with_summary.predict(input="你好")
    print(f"result: {result}")
    #此时我们调用memory 的 load_memory_variables 方法，可以看到记录下来的 history 是一小段关于对话的英文小结。
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(111111111111111111)
    result = conversation_with_summary.predict(input="请问鱼香肉丝怎么做")
    print(f"result: {result}")
    #结果：你可以继续问，这里的数据会英文内容小结会随着对话变多，说明每一次对话都在小结。
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")

    #原文https://blog.csdn.net/dfBeautifulLive/article/details/133350653?spm=1001.2014.3001.5501

def main6():
    SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。
    EXAMPLE
    Current summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。
    New lines of conversation:
    Human：为什么你认为人工智能是一种积极的力量？
    AI：因为人工智能将帮助人类发挥他们的潜能。
    New summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
    END OF EXAMPLE
    Current summary:
    {summary}
    New lines of conversation:
    {new_lines}
    New summary:"""

   # 总结的模板
    SUMMARY_PROMPT = PromptTemplate(
        input_variables=["summary", "new_lines"],
        template=SUMMARIZER_TEMPLATE
    )

    # 当对话的达到max_token_limit长度到多长之后，我们就应该调用 LLM 去把文本内容小结一下
    memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=256)

    CHEF_TEMPLATE = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文。
    2. 对于做菜步骤的回答尽量详细一些。
    {history}
    Human: {input}
    AI:"""

    CHEF_PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=CHEF_TEMPLATE
    )
    conversation_with_summar = ConversationChain(
        llm=OpenAI(stop="\n\n", max_tokens=2048, temperature=0.5),
        prompt=CHEF_PROMPT,
        memory=memory,
        verbose=True
    )
    result = conversation_with_summar.predict(input="你是谁？")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(111111111111111111)

    result = conversation_with_summar.predict(input="葱花饼怎么做？")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(222222222222222222)

    result = conversation_with_summar.predict(input="那发面饼怎么做呢？？")
    print(f"result: {result}")
    variables = memory.load_memory_variables({})
    print(f"variables: {variables}")
    print(3333333333333333)


    #原文https://blog.csdn.net/dfBeautifulLive/article/details/133350653?spm=1001.2014.3001.5501



if __name__ == "__main__":
    #main1()
    #main2()
    #main3()
    #main4()
    #main5()
    main6()
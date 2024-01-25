from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

examples = [
        {
            "question": "谁活得更久，穆罕默德·阿里还是艾伦·图灵？",
            "answer":
                """
                这里需要进一步的问题吗：是的。
                追问：穆罕默德·阿里去世时多大了？
                中间答案：穆罕默德·阿里去世时74岁。
                追问：艾伦·图灵去世时多大了？
                中间答案：艾伦·图灵去世时41岁。
                所以最终答案是：穆罕默德·阿里
                """
        },
        {
            "question": "craigslist的创始人是什么时候出生的？",
            "answer":
                """
                这里需要进一步的问题吗：是的。
                追问：谁是craigslist的创始人？
                中间答案：Craigslist是由Craig Newmark创办的。
                追问：Craig Newmark是什么时候出生的？
                中间答案：Craig Newmark出生于1952年12月6日。
                所以最终答案是：1952年12月6日
                """
        },
        {
            "question": "乔治·华盛顿的外祖父是谁？",
            "answer":
                """
                这里需要进一步的问题吗：是的。
                追问：谁是乔治·华盛顿的母亲？
                中间答案：乔治·华盛顿的母亲是Mary Ball Washington。
                追问：Mary Ball Washington的父亲是谁？
                中间答案：Mary Ball Washington的父亲是Joseph Ball。
                所以最终答案是：Joseph Ball
                """
        },
        {
            "question": "《大白鲨》和《皇家赌场》的导演是同一个国家的吗？",
            "answer":
                """
                这里需要进一步的问题吗：是的。
                追问：谁是《大白鲨》的导演？
                中间答案：《大白鲨》的导演是Steven Spielberg。
                追问：Steven Spielberg来自哪里？
                中间答案：美国。
                追问：谁是《皇家赌场》的导演？
                中间答案：《皇家赌场》的导演是Martin Campbell。
                追问：Martin Campbell来自哪里？
                中间答案：新西兰。
                所以最终答案是：不是
                """
        }
    ]
#使用 FewShotPromptTemplate 类生成 Few-shot Prompt
def main1():
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\n{answer}"
    )
    # **examples[0] 是将examples[0] 字典的键值对（question-answer）解包并传递给format，作为函数参数
    #** 用于将字典的键值对转换为函数的命名参数。
    # ** 用于字典解包。当您在函数调用中使用 ** 时，它会将字典的键值对作为关键字参数传递给函数。
    print(example_prompt.format(**examples[0]))

def main2():
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Question: {question}\n{answer}"
    )
    # **examples[0] 是将examples[0] 字典的键值对（question-answer）解包并传递给format，作为函数参数
    print(example_prompt.format(**examples[0]))
    print('1111111111')
    # 创建一个 FewShotPromptTemplate 对象
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,  # 使用前面定义的 examples 作为范例
        example_prompt=example_prompt,  # 使用前面定义的 example_prompt 作为提示模板
        suffix="Question: {input}",  # 后缀模板，其中 {input} 会被替换为实际输入
        input_variables=["input"]  # 定义输入变量的列表
    )
    # 使用给定的输入格式化 prompt，并打印结果
    # 这里的 {input} 将被 "玛丽·波尔·华盛顿的父亲是谁?" 替换
    prompt=few_shot_prompt.format(input="玛丽·波尔·华盛顿的父亲是谁?")
    print(prompt)
    print('2222222222222')
    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
    result = llm(prompt)
    print(f"result: {result}")
    print('333333333333')
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
    messages = [HumanMessage(content=prompt)]
    result = chat_model(messages)
    print(f"result: {result}")


def main3():
    # 定义一个提示模板
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],  # 输入变量的名字
        template="Input: {input}\nOutput: {output}",  # 实际的模板字符串
    )
    # 这是一个假设的任务示例列表，用于创建反义词
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    # 从给定的示例中创建一个语义相似性选择器
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,  # 可供选择的示例列表
        OpenAIEmbeddings(),  # 用于生成嵌入向量的嵌入类，用于衡量语义相似性
        Chroma,  # 用于存储嵌入向量并进行相似性搜索的 VectorStore 类
        k=1  # 要生成的示例数量
    )
   #**如果你有大量的参考示例，就得选择哪些要包含在提示中。最好还是根据某种条件或者规则来自动选择，Example Selector 是负责这个任务的类。**
    # 创建一个 FewShotPromptTemplate 对象
    similar_prompt = FewShotPromptTemplate(
        example_selector=example_selector,  # 提供一个 ExampleSelector 替代示例
        example_prompt=example_prompt,  # 前面定义的提示模板
        prefix="Give the antonym of every input",  # 前缀模板
        suffix="Input: {adjective}\nOutput:",  # 后缀模板
        input_variables=["adjective"],  # 输入变量的名字
    )
    print(similar_prompt.format(adjective="worried"))
    print('1111111111')
    print(similar_prompt.format(adjective="long"))
    print('2222222222')
    print(similar_prompt.format(adjective="rain"))
    print('333333333')

    prompt=similar_prompt.format(adjective="rain").format()
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=1000)
    messages = [HumanMessage(content=prompt)]
    result = chat_model(messages)
    print(f"result: {result}")




if __name__ == "__main__":
    # main1()
    #main2()
    main3()
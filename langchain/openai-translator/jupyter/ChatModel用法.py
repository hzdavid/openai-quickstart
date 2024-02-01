from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


def main1():
    chat_model = ChatOpenAI(model_name="gpt-4-1106-preview")
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    messages = [SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Who won the world series in 2020?"),
                AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
                HumanMessage(content="Where was it played?")]
    print(messages)
    print(chat_model(messages))
    return

def main2():
    # 翻译任务指令始终由 System 角色承担
    template = (
        """You are a translation expert, proficient in various languages. \n
        Translates English to Chinese."""
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    print(system_message_prompt)
    # 待翻译文本由 Human 角色输入
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    print(human_message_prompt)

    # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    print(chat_prompt_template)

    chat_prompt = chat_prompt_template.format_prompt(text="I love programming.").to_messages()
    # 为了翻译结果的稳定性，将 temperature 设置为 0
    translation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    result = translation_model(chat_prompt)
    print(result)
    return

def main3():

    # 无需再每次都使用 to_messages 方法构造 Chat Prompt
    # 为了翻译结果的稳定性，将 temperature 设置为 0
    # 翻译任务指令始终由 System 角色承担
    template = (
        """You are a translation expert, proficient in various languages. \n
        Translates English to Chinese."""
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    print(system_message_prompt)
    # 待翻译文本由 Human 角色输入
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    print(human_message_prompt)
    # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    print(chat_prompt_template)
    translation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    translation_chain = LLMChain(llm=translation_model, prompt=chat_prompt_template)
    # 等价于 translation_result.content (字符串类型)
    result = translation_chain.run({'text': "I love programming."})
    print(result)
    result =translation_chain.run({'text': "I love AI and Large Language Model."})
    print(result)
    return


def main4():

    # 无需再每次都使用 to_messages 方法构造 Chat Prompt
    # 为了翻译结果的稳定性，将 temperature 设置为 0
    # 翻译任务指令始终由 System 角色承担
    # System 增加 source_language 和 target_language
    template = (
        """You are a translation expert, proficient in various languages. \n
        Translates {source_language} to {target_language}."""
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    print(system_message_prompt)
    # 待翻译文本由 Human 角色输入
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    print(human_message_prompt)

    # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
    m_chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )





    # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    print(chat_prompt_template)
    translation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    translation_chain = LLMChain(llm=translation_model, prompt=m_chat_prompt_template)
    # 等价于 translation_result.content (字符串类型)
    result =translation_chain.run({
    "source_language": "Chinese",
    "target_language": "English",
    "text": "我喜欢学习大语言模型，轻松简单又愉快",})
    print(result)
    result = translation_chain.run({
        "source_language": "Chinese",
        "target_language": "Japanese",
        "text": "我喜欢学习大语言模型，轻松简单又愉快", })
    print(result)

    return




if __name__ == "__main__":
    #main1()
    #main2()
    #main3()
    main4()

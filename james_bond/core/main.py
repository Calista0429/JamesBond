from dotenv import load_dotenv
from llm import BondAgentLLM
# 加载环境变量

if __name__ == "__main__":
    load_dotenv()
    llm = BondAgentLLM()

    # 准备消息
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

    # 发起调用，think等方法都已从父类继承，无需重写
    response_stream = llm.think(messages)

    # 打印响应
    print("Model Response:")
    for chunk in response_stream:
        # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
        # print(chunk, end="", flush=True)
        pass
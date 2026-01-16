import os
from openai import OpenAI


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    # 以下为新加坡地域base_url，若使用北京地域的模型，需将base_url替换为：https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key="sk-137ad80ef0494cacb4e1e70c3fafc78c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  
)

completion = client.chat.completions.create(
    # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
    model="qwen-flash",
    messages=[
        {"role": "user", "content": "How does the United States reconcile its support for the One China policy with its defense commitments under the Taiwan Relations Act?"}
    ],
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    # extra_body={"enable_thinking": False},
)
print(completion.model_dump_json())

# export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
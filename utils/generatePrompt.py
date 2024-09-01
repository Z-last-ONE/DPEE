
import json
import openai
import numpy as np
import logging
import os
import time

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

openai.api_key = "YOUR OPENAI API_KEY"

def get_gpt_response_w_system(prompt):
    global system_prompt
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    return response


# read the system_prompt (Instruction) for item profile generation
system_prompt = ""
with open('./event_prompt.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        system_prompt += line


# read the example prompts of items
# example_prompts = []
# with open('D:/firstproject/opportunity/shiyan/testInputPrompt.json', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         i_prompt = json.loads(line)
#         example_prompts.append(i_prompt['prompt'])

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

# print(Colors.GREEN + "Generating Profile for Item" + Colors.END)
# print("---------------------------------------------------\n")
# print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
# print(system_prompt)
# print("---------------------------------------------------\n")
# print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
# print(example_prompts[700])
# print("---------------------------------------------------\n")
# start_time = time.time()
# response = get_gpt_response_w_system(example_prompts[700])
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"程序运行时间：{execution_time} 秒")
# print(Colors.GREEN + "Generated Results:\n" + Colors.END)
# print(response)

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,  # 设置最低的日志级别为DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trainSeven.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('my_logger')


# 批处理所有的提示数据
def chat_with_LLM():
    reasoning_result = []
    try:
        start_time = time.time()
        with open("./trainInputPrompt.json", 'r', encoding='utf-8') as file:
            dataset = [json.loads(x) for x in file.readlines()]
            for i, data in enumerate(dataset[7044:7185]):
                logger.info('执行到第%d个数据', i)
                logger.info('执行的数据id为%s', data['id'])
                prompt = data['prompt']
                result = get_gpt_response_w_system(prompt)
                logger.info('程序输出结果为:%s', result)
                tempDict={}
                tempDict['id'] = data["id"]
                tempDict['result']=result
                reasoning_result.append(tempDict)
                with open('trainSeven1.json', 'w', encoding='utf-8') as file:
                    for i, item in enumerate(reasoning_result):
                        json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为JSON字符串
                        file.write(json_line + '\n')
            # with open('result.json', 'w', encoding='utf-8') as f:
            #     json.dump(reasoning_result, f, ensure_ascii=False)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"程序运行时间：{execution_time} 秒")
    except Exception as e:
        logger.error('处理过程中发生错误: %s', e, exc_info=True)

if __name__ == "__main__":
    chat_with_LLM()

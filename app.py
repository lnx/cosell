import os
from enum import Enum

from openai import OpenAI

SYSTEM_PROMPTS = [
    "你是一个经验丰富且耐心的客服，对于用户的问题都能热情并简短的给出答复。请尽可能的简短一些，专注于解决用户的问题，不要啰嗦。",
    "所有回答的内容请以如下这段内容为准，回答所使用的内容如下：",
]

GREETINGS = '请问有什么可以帮您？'


class Role(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class Message:
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content


def load_product_info(customer: str):
    return ''.join(open('data/{}.jsonl'.format(customer), 'r').readlines())


client = OpenAI(api_key=os.getenv("MOONSHOT_API_KEY"), base_url="https://api.moonshot.cn/v1")


def chat(messages: list[Message]):
    return client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[m.__dict__ for m in messages],
        temperature=0.3
    ).choices[0].message


if __name__ == '__main__':
    messages = [
        Message(Role.SYSTEM, ''.join(SYSTEM_PROMPTS)),
        Message(Role.SYSTEM, load_product_info('yili')),
    ]

    print('>>>> 客服：\n' + GREETINGS)
    while True:
        print('\n>>>> 我：')
        messages.append(Message(Role.USER, input()))
        m = chat(messages)
        print('\n>>>> 客服：\n' + m.content)
        messages.append(Message(Role.ASSISTANT, m.content))

import openai

APIKEYS=[
    ""
]
 
GPT4_APIKEYS=[
   ""
]

def _check_key(message: str, key='', model='gpt-3.5-turbo') -> str:
    """
    model: gpt-3.5-turbo, gpt-3.5-turbo-0301, gpt-4
    """
    message_log = [{"role": "user", "content": message}]
    openai.api_key = key

    completion = openai.ChatCompletion.create(model=model,
                                               messages=message_log,
                                               max_tokens=1)
    res = completion.choices[0].message.content

    print(f"key:{key}, question: {message} ===> answer: {res}")
    return res


if __name__=='__main__':
    for key in APIKEYS:
        datas = 'hello'
        _check_key(datas, key)
    print('gpt3.5 keys are fine')

    for key in GPT4_APIKEYS:
        print(key)
        datas = 'hello'
        _check_key(datas, key, model='gpt-4')
         
    print('gpt4 keys are fine')
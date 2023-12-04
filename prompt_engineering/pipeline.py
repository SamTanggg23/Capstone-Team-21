from api_utils import query_chatgpt_and_save_results
from apikeys import APIKEYS, GPT4_APIKEYS
import os
import tiktoken

class BasePipeline():
    def __init__(self, template, parse_func):
        self.template = template
        self.parse_func = parse_func

    def num_tokens_from_string(self, string: str, model_name: str):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens, encoding

    def build_prompt(self, id, input_text, model_name="", truncate=False, truncate_nums=4096):
        if truncate:
            template_token_nums, encoding = self.num_tokens_from_string(self.template, model_name=model_name)
            input_tokens = encoding.encode(input_text)
            need_nums = truncate_nums-template_token_nums-20
            if need_nums < len(input_tokens):
                input_tokens_trunc = input_tokens[:need_nums]
                trunc_text = encoding.decode(input_tokens_trunc)
                trunc_text = trunc_text[:-3]
                assert trunc_text in input_text
            else:
                trunc_text = input_text

        else:
            trunc_text = input_text
        return {'id': id, 'prompt': self.template.format(trunc_text)}


    def parse(self, raw_text):
        return self.parse_func(raw_text)

    def post_func(self, gpt_response):
        try:
            if 'output' in gpt_response:
                return gpt_response # already parsed
            data = self.parse(raw_text=gpt_response['response_metadata'])
        except Exception as e:
            print(f'parsing error ({e}), here is the raw response: ...............')
            print(gpt_response)
            return None
            # raise e
        return {'output': data, 'id':gpt_response['id'], "raw_response": gpt_response['response_metadata']}

    def query(
        self,
        texts,
        engine, 
        gpt_outputs_path=None, 
        num_processes=10, 
        retry_limit=5, 
        truncate=False,
        truncate_nums=4096,
        **completion_kwargs):
        if engine == 'gpt-4':
            apikeys = GPT4_APIKEYS
        else:
            apikeys = APIKEYS
        prompts = [self.build_prompt(i,t, engine, truncate, truncate_nums) for i,t in enumerate(texts)]
        assert self.template is not None
        assert self.parse_func is not None
        return query_chatgpt_and_save_results(
            apikeys=apikeys,
            engine=engine,
            instances_generator=prompts,
            instances_number=len(prompts),
            post_function=self.post_func,
            output_path=gpt_outputs_path,
            num_processes=num_processes,
            retry_limit=retry_limit,
            **completion_kwargs
        )
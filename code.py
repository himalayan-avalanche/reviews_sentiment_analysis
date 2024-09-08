import openai, os, pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
import json
import time

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)


_=load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model_3 = "gpt-3.5-turbo"    
llm_model_4='gpt-4'    


def get_response_details(customer_review, model):
    
    from langchain.chat_models import ChatOpenAI
    
    review_template = """For the following text, extract the following information and output as python dictionary:

                        kwd:              Summarize the customer review with top keywords that express customer sentiment and \
                                          useful improving the customer experience and business operations.

                        summary_review:   Summarize the review in 1 sentence, max 20 words.

                        biz_response:     Write a appropriate response to the customer review, in american english and \
                                          respectful tone.


                        text: {text}

                        """
        
    prompt = ChatPromptTemplate.from_template(template=review_template)
    
    messages = prompt.format_messages(text=customer_review)
    
    chat_model= ChatOpenAI(temperature=0.2, model=model)
    
    response = chat_model(messages)

    return response.content


def get_formatted_response(res):
    return json.loads(res)


#### Calls to LLMs GPT Turbo 3.5 and GPT 4

st=time.time()

res_11=get_response_details(review_text_1,model=llm_model_3)
res_12=get_response_details(review_text_2,model=llm_model_3)
res_21=get_response_details(review_text_1,model=llm_model_4)
res_22=get_response_details(review_text_2,model=llm_model_4)

en=time.time()

print(f'time take {en-st}')

res=pd.DataFrame()
res['review_text']=[review_text_1,review_text_2]
res['kwd_gpt_3.5']=[get_formatted_response(res_11)['kwd'],get_formatted_response(res_12)['kwd']]
res['kwd_gpt_4']=[get_formatted_response(res_21)['kwd'],get_formatted_response(res_22)['kwd']]
res['summary_review_gpt_3.5']=[get_formatted_response(res_11)['summary_review'],get_formatted_response(res_12)['summary_review']]
res['summary_review_gpt_4']=[get_formatted_response(res_21)['summary_review'],get_formatted_response(res_22)['summary_review']]
res['biz_response_gpt_3.5']=[get_formatted_response(res_11)['biz_response'],get_formatted_response(res_12)['biz_response']]
res['biz_response_gpt_4']=[get_formatted_response(res_21)['biz_response'],get_formatted_response(res_22)['biz_response']]

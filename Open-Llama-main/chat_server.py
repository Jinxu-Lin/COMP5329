"""
Author: s-JoL(sl12160010@gmail.com)
Date: 2023-04-06 22:30:10
LastEditors: s-JoL(sl12160010@gmail.com)
LastEditTime: 2023-05-12 15:07:36
FilePath: /Open-Llama/chat_server.py
Description: 

Copyright (c) 2023 by s-JoL(sl12160010@gmail.com), All Rights Reserved. 
"""
import torch
import logging
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("s-JoL/Open-Llama-V2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "s-JoL/Open-Llama-V2", torch_dtype=torch.bfloat16, device_map="auto"
)
logging.warning("ready")


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # [Open-Llama](https://github.com/s-JoL/Open-Llama)
    完全使用Open-Llama项目从0开始训练的Instruct-GPT模型，当长时间无响应（如20s以上）可刷新重试。

    Instruct-GPT model is trained from scratch using the Open-Llama project without relying on any other pre-trained models. If there is no response for a long time (such as more than 20 seconds), please refresh and try again. 
    """
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        logging.warning(user_message)
        return "", history + [[user_message, None]]

    def bot(history):
        context = []
        round = 0
        for prompt, completion in history:
            round += 1
            if completion is None:
                inputs = "user:{}\nsystem:".format(prompt)
                inputs = tokenizer(
                    inputs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                context.append(inputs["input_ids"])
            else:
                inputs = "user:{}\nsystem:{}".format(prompt, completion)
                inputs = tokenizer(
                    inputs,
                    return_tensors="pt",
                    add_special_tokens=True,
                    return_attention_mask=False,
                )
                context.append(inputs["input_ids"])
        context = torch.cat(context, dim=-1)
        context = context[:, -1024:]
        inputs_len = context.shape[1]
        context = context.cuda()
        pred = model.generate(input_ids=context, max_new_tokens=1024, do_sample=True)
        pred = pred[:, inputs_len:]
        pred = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        logging.warning(pred)
        bot_message = pred
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    gr.Markdown(
        """
    当前体验服务生成的所有内容都是由人工智能模型生成，我们对其生成内容的准确性、完整性和功能性不做任何保证，并且其生成的内容不代表我们的态度或观点。

    联系方式: sl12160010@gmail.com  对于该项目有任何意见和建议都欢迎联系我.
    
    Contact information: sl12160010@gmail.com. Any opinions or suggestions regarding the project are welcome to be addressed to me through this email.
    """
    )

demo.launch()

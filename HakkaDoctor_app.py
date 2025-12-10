#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()
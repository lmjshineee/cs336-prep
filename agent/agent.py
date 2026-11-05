import json
import os
import subprocess
import sys

from openai import OpenAI

client = OpenAI(
    api_key = os.environ[""],
    base_url = os.environ.get("MODEL", "http://api.deepseek.com"),
)
MODEL = os.environ.get("MODEL", "deepseek-flash")
MAX_STEPS = 20

SYSTEM = (
    "你是一个能操作本地文件和命令行的编程助手。"
    "需要时调用工具，完成任务后直接用文字总结，不要再调用工具。"
)

def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path} {len(content) 字符} "

def run_shell(command: str) -> str:
    if input(f"\n[需要确认] 执行命令： {command} ")

"""Code evaluation module."""
from __future__ import annotations

from autogpt.commands.command import command
from autogpt.llm import call_ai_function
from autogpt.llm.base import Message
from typing import List
from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion


@command(
    "decompose",
    "Decompose the task",
    '"task": "<copy_full_task_description>"',
)
def decompose(task: str) -> str:
    """
    A function that takes in a string and returns a response from create chat
      completion api call.

    Parameters:
        code (str): Code to be evaluated.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """

    cfg = Config()
    model = cfg.smart_llm_model

    # Write a list of tasks
    prompt = f"""You need to write a list of tasks that together is a tutorial that describe subtasks and commands you need to do. 
Usage:Write a list of subtasks for this task according to best practices starting from 1., to fulfill the final request of the user and taking into account the instructions and the previous task.

Example:
Task / objective: Write a streamlit application with openai chatbot functionality
The goals you need to achieve in the tutorial to complete the task:
1. Familiarize yourself with Streamlit and OpenAI chatbot API.
2. Install the necessary packages and dependencies for Streamlit and OpenAI chatbot API.
3. Write a Python code for a Streamlit application that integrates OpenAI chatbot.
4. Test the code thoroughly and debug any issues that arise.
5. Incorporate error handling and other features to improve the robustness and reliability of the application.
6. Document the code and provide clear instructions for how to run and use the application.

Task / objective: {task}"""
    messages: List[Message] = [
        {
            "role": "system",
            "content": f"You are now the expert tutorial and manual writer",
        },
        {"role": "user", "content": prompt},
    ]
    return create_chat_completion(model=model, messages=messages, temperature=0)


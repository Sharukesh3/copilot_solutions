import re

def remove_think_tags(input_string):
    """
    Removes the content between <think> and </think> tags, including the tags themselves.

    Args:
        input_string (str): The input string.

    Returns:
        str: The string with <think>...</think> content removed.
    """
    return re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
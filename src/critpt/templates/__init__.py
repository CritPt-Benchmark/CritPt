
from critpt.templates.templates import ParsePrompt, SystemPrompt


def get_default_system_prompt(_parsing):
    prompt_style = "one-step" if _parsing else "two-step"
    return SystemPrompt.default_system_prompt(prompt_style)

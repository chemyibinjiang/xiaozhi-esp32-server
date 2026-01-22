STATUS_PREFIX = "[[status]]"


def wrap_status(text: str) -> str:
    if not text:
        return ""
    return f"{STATUS_PREFIX}{text}"


def extract_status(content: str):
    if not content:
        return None
    if content.startswith(STATUS_PREFIX):
        return content[len(STATUS_PREFIX):].lstrip()
    return None

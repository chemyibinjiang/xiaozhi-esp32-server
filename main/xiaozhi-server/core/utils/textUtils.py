import json
import re

TAG = __name__

EMOJI_MAP = {
    "😀": "happy",
    "😃": "happy",
    "😄": "happy",
    "😁": "funny",
    "😆": "funny",
    "😊": "loving",
    "😍": "loving",
    "😘": "kissy",
    "😎": "cool",
    "😢": "crying",
    "😭": "crying",
    "😠": "angry",
    "😡": "angry",
    "😮": "surprised",
    "😱": "shocked",
    "🤔": "thinking",
    "😴": "sleepy",
    "😜": "silly",
    "😕": "confused",
    "😐": "neutral",
    "😳": "embarrassed",
    "😉": "winking",
    "😋": "delicious",
    "😌": "relaxed",
    "😏": "confident",
    "😞": "sad",
}

EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
]

PUNCTUATION_SET = {
    ",",
    ".",
    "!",
    "?",
    ":",
    ";",
    "-",
    "~",
    "[",
    "]",
    "(",
    ")",
    "，",
    "。",
    "！",
    "？",
    "：",
    "；",
    "（",
    "）",
    "【",
    "】",
    "\"",
    "'",
    "“",
    "”",
    "‘",
    "’",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;\n])")

BACKSTAGE_PROCESS_KEYWORDS = (
    "读取",
    "记录",
    "写入",
    "补全",
    "校验",
    "切到下一步",
    "进入下一步",
    "推进到下一步",
    "读取当前步骤",
    "读取当前子步骤",
    "读取动作说明",
    "读取唯一动作",
    "确认当前步骤",
    "确认当前子步骤",
    "切换到当前步骤",
    "切换到当前子步骤",
    "建立会话",
    "会话",
    "工具",
    "调用",
    "后台",
    "检索",
    "加载",
    "恢复记录",
    "导出",
    "拍照确认",
    "强制拍照确认",
    "只给你当前动作",
    "只给你一个动作",
    "只告诉你当前要做的一个动作",
    "只告诉你这一句操作指令",
    "当前要做的一个动作",
    "动作说明",
    "操作指令",
    "下一句操作指令",
)

BACKSTAGE_PREFIXES = (
    "收到你的完成反馈",
    "收到",
    "继续",
    "我先",
    "我会先",
    "我现在",
    "我这边先",
    "我这边现在",
    "我继续",
    "我马上",
    "正在",
    "先帮你",
    "先给你",
    "接着",
    "然后",
    "马上",
)

BACKSTAGE_LEADING_PATTERNS = [
    re.compile(r"^(?:当前要做的一个动作|给你当前要做的一个动作)[。！？；，、,\s]*"),
    re.compile(
        r"^收到[^。！？；]{0,40}(?:完成反馈|反馈|结果)[^。！？；]{0,20}[，、,\s]*"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我这边先|我这边现在|我继续|我马上|正在|先帮你|先给你)"
        r"[^。！？；]{0,120}"
        r"(?:记录|写入|补全|校验|切到下一步|进入下一步|推进到下一步|读取当前步骤|读取当前子步骤|"
        r"读取动作说明|读取唯一动作|确认当前步骤|确认当前子步骤|切换到当前步骤|切换到当前子步骤|"
        r"建立会话|工具|调用|后台|检索|加载|恢复记录|导出|拍照确认|强制拍照确认|"
        r"只给你当前动作|只给你一个动作|只告诉你当前要做的一个动作|只告诉你这一句操作指令|"
        r"当前要做的一个动作|动作说明|操作指令|下一句操作指令)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^(?:接着|然后|马上)"
        r"[^。！？；]{0,120}"
        r"(?:切到下一步|进入下一步|推进到下一步|只给你当前动作|只给你一个动作|"
        r"只告诉你当前要做的一个动作|只告诉你这一句操作指令|当前要做的一个动作|"
        r"动作说明|操作指令|下一句操作指令)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^继续"
        r"[^。！？；]{0,120}"
        r"(?:记录|校验|确认下一步动作|当前步骤|当前子步骤|拍照确认|推进)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
]

BACKSTAGE_FILLER_PATTERNS = [
    re.compile(r"^(?:继续|收到|好的|好)[。！？；，、,\s]*$"),
]

BACKSTAGE_REWRITE_PATTERNS = [
    (
        re.compile(
            r"^拍照[^。！？；]{0,120}(?:路由冲突|会话路由冲突|重试处理)[^。！？；]{0,120}[。！？；，、,\s]*$"
        ),
        "拍照暂时没成功，请稍后再试。",
    ),
    (
        re.compile(
            r"^现在请在设备端断开并重新进入一次实验会话[^。！？；]{0,120}[。！？；，、,\s]*$"
        ),
        "拍照暂时没成功，请稍后再试。",
    ),
]


def get_string_no_punctuation_or_emoji(s):
    """Strip leading/trailing whitespace, punctuation, and emoji."""
    chars = list(s or "")
    start = 0
    while start < len(chars) and is_punctuation_or_emoji(chars[start]):
        start += 1

    end = len(chars) - 1
    while end >= start and is_punctuation_or_emoji(chars[end]):
        end -= 1

    return "".join(chars[start : end + 1])


def is_punctuation_or_emoji(char):
    if char.isspace() or char in PUNCTUATION_SET:
        return True
    return is_emoji(char)


async def get_emotion(conn, text):
    """Send a simple emotion hint based on the first emoji found in text."""
    emoji = "😃"
    emotion = "happy"
    for char in text or "":
        if char in EMOJI_MAP:
            emoji = char
            emotion = EMOJI_MAP[char]
            break
    try:
        await conn.websocket.send(
            json.dumps(
                {
                    "type": "llm",
                    "text": emoji,
                    "emotion": emotion,
                    "session_id": conn.session_id,
                }
            )
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"发送情绪表情失败，错误:{e}")


def is_emoji(char):
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in EMOJI_RANGES)


def check_emoji(text):
    """Remove emoji and newlines from text before it is spoken."""
    return "".join(char for char in (text or "") if not is_emoji(char) and char != "\n")


def normalize_spoken_text(text):
    """Deterministic spoken-text normalization: collapse whitespace only."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _strip_backstage_leading_clauses(text: str) -> str:
    cleaned = (text or "").strip()
    while cleaned:
        updated = cleaned
        for pattern in BACKSTAGE_LEADING_PATTERNS:
            updated = pattern.sub("", updated, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _looks_like_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if not any(keyword in stripped for keyword in BACKSTAGE_PROCESS_KEYWORDS):
        return False
    return stripped.startswith(BACKSTAGE_PREFIXES)


def _is_backstage_filler_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in BACKSTAGE_FILLER_PATTERNS)


def _rewrite_backstage_sentence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    for pattern, replacement in BACKSTAGE_REWRITE_PATTERNS:
        if pattern.fullmatch(stripped):
            return replacement
    return stripped


def filter_spoken_backstage_text(text):
    """Drop backend workflow narration while preserving student-facing instructions."""
    if not text:
        return text

    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return ""

    # Hard opening rule: if the canonical opening sentence exists, keep only that sentence.
    opening_match = re.search(r"今天我们做《[^》\n]{1,80}》。你准备好开始了吗[？?]", normalized)
    if opening_match:
        return opening_match.group(0)

    sentence_split_re = re.compile(r"(?<=[。！？!?；;])\s*")
    backstage_prefixes = (
        "我先",
        "我再",
        "我会",
        "我现在",
        "我继续",
        "先帮你",
        "接着",
        "然后",
        "马上",
        "正在",
        "会话已",
        "我这边",
    )
    backstage_keywords = (
        "后台",
        "会话",
        "读取",
        "读一下",
        "记录",
        "写入",
        "校验",
        "工具",
        "调用",
        "当前步骤",
        "当前子步骤",
        "切到下一步",
        "推进到下一步",
        "检索",
        "加载",
    )
    filler_re = re.compile(r"^(收到|好的|好|明白了?|嗯|继续)[，,。！？!?；; ]*$")

    def should_drop_sentence(sentence: str) -> bool:
        s = sentence.strip()
        if not s:
            return True
        if filler_re.fullmatch(s):
            return True

        has_backstage_keyword = any(k in s for k in backstage_keywords)
        starts_with_backstage_prefix = s.startswith(backstage_prefixes)
        if has_backstage_keyword and starts_with_backstage_prefix:
            return True

        # Also drop explicit backend narration even without the common prefixes.
        if "并先在后台" in s or "再读取当前步骤" in s or "读取实验概览" in s:
            return True
        return False

    kept_sentences = []
    for raw_sentence in sentence_split_re.split(normalized):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if should_drop_sentence(sentence):
            continue
        if not get_string_no_punctuation_or_emoji(sentence):
            continue
        if kept_sentences and kept_sentences[-1] == sentence:
            continue
        kept_sentences.append(sentence)

    if kept_sentences:
        # Keep response concise to reduce TTS fragmentation.
        return "".join(kept_sentences[:2]).strip()

    # Fallback: only remove a leading backstage clause, keep the rest.
    leading_clause_re = re.compile(
        r"^(?:我先|我再|我会|我现在|我继续|接着|然后|马上|正在|我这边)"
        r"[^。！？!?；;]{0,120}"
        r"(?:后台|会话|读取|记录|写入|校验|工具|调用|当前步骤|切到下一步|推进到下一步)"
        r"[^。！？!?；;]{0,120}[，,\s]*"
    )
    cleaned = leading_clause_re.sub("", normalized).strip()
    return cleaned or normalized

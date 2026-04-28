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
    "我把",
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
    "这一步是",
    "这一步先是",
)

BACKSTAGE_LEADING_PATTERNS = [
    re.compile(r"^(?:当前要做的一个动作|给你当前要做的一个动作)[。！？；，、,\s]*"),
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:总览确认|总览说明|总览阶段|概览确认)"
        r"[^。！？；]{0,160}"
        r"(?:记录什么|进入共同准备阶段|进入共同准备的第一步|开始共同准备阶段|带你进入)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^收到[^。！？；]{0,40}(?:完成反馈|反馈|结果)[^。！？；]{0,20}[，、,\s]*"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在|我继续|我马上|正在|先帮你|先给你)"
        r"[^。！？；]{0,120}"
        r"(?:记录|记上|写上|写入|补全|校验|切到下一步|进入下一步|推进到下一步|推进到|带你进入|"
        r"进入共同准备阶段|进入共同准备的第一步|"
        r"读取当前步骤|读取当前子步骤|"
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
            r"^这一步(?:先)?是[^。！？；]{0,120}"
            r"(?:总览确认|总览说明|总览阶段|概览确认)"
            r"[^。！？；]{0,160}[。！？；，、,\s]*$"
        ),
        "先确认整体安排，准备好后就开始共同准备阶段。",
    ),
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

BACKSTAGE_FULL_SENTENCE_PATTERNS = [
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:总览确认|总览说明|总览阶段|概览确认)"
        r"[^。！？；]{0,160}"
        r"(?:记录什么|带你进入|共同准备阶段|共同准备的第一步)"
        r"[^。！？；]{0,120}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在)"
        r"[^。！？；]{0,200}"
        r"(?:记上|写上|写入|记录|推进到|带你进入|进入共同准备阶段|进入共同准备的第一步|进入下一步|切到下一步)"
        r"[^。！？；]{0,200}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在)"
        r"[^。！？；]{0,200}"
        r"(?:记上|写上|写入|记录)"
        r"[^。！？；]{0,120}"
        r"(?:带你做下一步|下一步共同加液|下一步共同操作|下一步共同准备)"
        r"[^。！？；]{0,160}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:我先查一下|我再查一下|我继续补一下|我补一下|我继续确认|我先确认)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^这一步需要(?:把)?[^。！？；]{0,160}"
        r"(?:我继续补一下|我补一下|我先查一下|我继续确认|尽量把你现在要量的内容说准确|避免你按错量做)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^现在(?:开始|进入)当前步骤[^。！？；]{0,120}"
        r"(?:我先|我会先|我现在)"
        r"[^。！？；]{0,200}"
        r"(?:确认|动作和记录要求|只带你做这一小步|只给你这一步|当前动作)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?我先查这一步在实验说明里的具体配法(?:，|,)?只告诉你现在要配的这一项[。！？；，\s]*$"
    ),
    re.compile(
        r"^这一步的浓度和用法我查到了(?:，|,)?我再核对一下实验说明里有没有写明具体配制量[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:只需要记|这一步只需要记)[^。！？；]{0,120}(?:不往这一步里乱写|不乱写)[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?我进入(?:[一二三四五]|[1-5])号样品(?:，|,)?只讲(?:[一二三四五]|[1-5])号现在该加什么[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我现在|我继续)?(?:切到|进入)[^。！？；]{0,80}"
        r"(?:拍照这一步|拍照确认这一步|强制拍照这一步)"
        r"[^。！？；]{0,80}[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我现在|我这边现在)?执行(?:[一二三四五]|[1-5])号样品的(?:强制)?拍照"
        r"(?:，|,)?并把拍照结果写回当前步骤[。！？；，\s]*$"
    ),
    re.compile(
        r"^拍照已经成功(?:，|,)?我把拍照确认记录写回当前步骤并准备进入(?:[一二三四五]|[1-5])号样品[。！？；，\s]*$"
    ),
]


STRUCTURAL_BACKSTAGE_PATTERNS = [
    re.compile(
        "^(?:\\u6211\\u6309\\u4f60|\\u53ea\\u9700\\u8981\\u8bb0|\\u8fd9\\u4e00\\u6b65\\u53ea\\u9700\\u8981\\u8bb0)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u5199\\u5165|\\u8bb0\\u4e0a|\\u8bb0\\u5230|\\u4e0d\\u8865|\\u4e0d\\u5f80\\u8fd9\\u4e00\\u6b65\\u91cc\\u4e71\\u5199|\\u4e0d\\u4e71\\u5199)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u518d|\\u6211\\u5148|\\u6211\\u4f1a|\\u6211\\u73b0\\u5728|\\u6211\\u7ee7\\u7eed)"
        "(?:\\u786e\\u8ba4\\u4e00\\u4e0b|\\u786e\\u8ba4|\\u6838\\u5bf9\\u4e00\\u4e0b|\\u6838\\u5bf9)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5f55\\u5b57\\u6bb5|\\u8bb0\\u5f55\\u8981\\u6c42|\\u5b57\\u6bb5)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,220}"
        "(?:\\u8865\\u62a5|\\u6f0f\\u62a5|\\u522b\\u7684\\u4fe1\\u606f|\\u522b\\u7684\\u5185\\u5bb9)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fd9\\u4e00\\u6b65\\u8bb0\\u5f55\\u9f50\\u4e86"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u7ed3\\u675f\\u5f53\\u524d\\u6b65\\u9aa4|\\u5207\\u5230\\u4e0b\\u4e00\\u6b65|\\u5171\\u540c\\u52a8\\u4f5c|\\u63a5\\u4e0b\\u6765\\u8981\\u505a\\u7684\\u5171\\u540c\\u52a8\\u4f5c)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u5148\\u628a(?:\\u8fd9\\u4e00\\u8f6e|\\u5f53\\u524d|\\u8fd9\\u4e00\\u6b65)?"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5f55\\u5b8c\\u6210|\\u8bb0\\u5f55\\u505a\\u5b8c|\\u8bb0\\u5f55\\u9f50\\u4e86|\\u8bb0\\u5f55\\u7ed3\\u675f)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u518d\\u8fdb\\u5165|\\u518d\\u5207\\u5230|\\u7136\\u540e\\u8fdb\\u5165)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u4e0b\\u4e00\\u79cd\\u5171\\u540c\\u8bd5\\u5242|\\u4e0b\\u4e00\\u79cd\\u8bd5\\u5242|\\u5171\\u540c\\u8bd5\\u5242|\\u4e0b\\u4e00\\u8f6e\\u5171\\u540c\\u52a0\\u6db2|\\u5171\\u540c\\u52a8\\u4f5c)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u7ee7\\u7eed)"
        "\\u628a[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u8bb0\\u4e0a|\\u5199\\u5165|\\u8bb0\\u5230|\\u5199\\u5230)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5b8c\\u6210\\u62cd\\u7167\\u786e\\u8ba4|\\u62cd\\u7167\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u4e0b\\u4e00\\u6b65\\u662f[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f3a\\u5236\\u62cd\\u7167\\u786e\\u8ba4|\\u62cd\\u7167\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u6211\\u76f4\\u63a5\\u6267\\u884c|\\u6211\\u73b0\\u5728\\u6267\\u884c|\\u6211\\u9a6c\\u4e0a\\u6267\\u884c)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u62cd\\u597d\\u4e86(?:\\uff0c|,)?(?:\\u6211\\u628a)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u62cd\\u7167\\u786e\\u8ba4|\\u7167\\u7247\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5230|\\u8bb0\\u4e0a|\\u5199\\u5230|\\u5199\\u5165)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?\\u6211\\u8fdb\\u5165(?:[\\u4e00\\u4e8c\\u4e09\\u56db\\u4e94]|[1-5])\\u53f7\\u6837\\u54c1(?:\\uff0c|,)?"
        "(?:\\u53ea\\u8bb2|\\u53ea\\u7ed9\\u4f60)[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u8f6e|\\u73b0\\u5728\\u8be5\\u52a0\\u4ec0\\u4e48|\\u4f53\\u79ef)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?\\u6211\\u5148\\u67e5[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5b9e\\u9a8c\\u8bf4\\u660e|\\u6587\\u6863)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u5177\\u4f53\\u914d\\u6cd5|\\u914d\\u5236\\u91cf|\\u8fd9\\u4e00\\u9879)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fd9\\u4e00\\u6b65\\u7684\\u6d53\\u5ea6\\u548c\\u7528\\u6cd5\\u6211\\u67e5\\u5230\\u4e86(?:\\uff0c|,)?"
        "\\u6211\\u518d\\u6838\\u5bf9\\u4e00\\u4e0b(?:\\u5b9e\\u9a8c\\u8bf4\\u660e|\\u6587\\u6863)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u5177\\u4f53\\u914d\\u5236\\u91cf|\\u5177\\u4f53\\u914d\\u6cd5)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u73b0\\u5728|\\u6211\\u5148|\\u6211\\u518d)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u542f\\u52a8|\\u5f00\\u59cb)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u6279\\u91cf\\u626b\\u63cf|\\u626b\\u63cf)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u53c2\\u6570\\u683c\\u5f0f\\u4e0d\\u5bf9(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u91cd\\u8bd5|\\u6309\\u6b63\\u786e\\u683c\\u5f0f\\u91cd\\u8bd5)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6279\\u91cf\\u626b\\u63cf|\\u626b\\u63cf)[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "\\u8d85\\u65f6\\u4e86(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u72b6\\u6001|\\u91cd\\u8bd5)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u626b\\u63cf\\u8bf7\\u6c42\\u5df2\\u7ecf\\u53d1\\u51fa|\\u8bf7\\u6c42\\u5df2\\u7ecf\\u53d1\\u51fa)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u7ed3\\u679c\\u8fd8\\u6ca1\\u8fd4\\u56de|\\u8fd8\\u6ca1\\u8fd4\\u56de)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u6211\\u7b49\\u4e00\\u4f1a\\u513f\\u518d\\u67e5\\u4e00\\u6b21[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u73b0\\u5728|\\u6211\\u518d|\\u6211\\u5148)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u91cd\\u65b0\\u67e5\\u8be2|\\u67e5\\u8be2)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u626b\\u63cf\\u6709\\u6ca1\\u6709\\u7ed3\\u675f|\\u6709\\u6ca1\\u6709\\u7ed3\\u675f)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fde\\u63a5[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u65ad\\u4e86\\u4e00\\u4e0b|\\u65ad\\u5f00\\u4e86)"
        "(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u91cd\\u8fde\\u5149\\u8c31\\u4eea|\\u786e\\u8ba4\\u72b6\\u6001)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9)?\\u6682\\u65f6\\u53d6\\u4e0d\\u5230(?:\\u626b\\u63cf\\u7ed3\\u679c|\\u7ed3\\u679c)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
]

TRANSITION_BACKSTAGE_PATTERNS = [
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728|"
        "\\u6211\\u8fd9\\u8fb9\\u5148|\\u6211\\u8fd9\\u8fb9\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u5e26\\u4f60\\u505a|\\u5e26\\u4f60\\u8fdb|\\u5e26\\u4f60\\u7ee7\\u7eed|"
        "\\u7ee7\\u7eed\\u5e26\\u4f60|\\u53ea\\u7ed9\\u4f60\\u8fd9\\u4e00\\u6b65|"
        "\\u53ea\\u5e26\\u4f60\\u505a)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d|\\u8fd9\\u4e00\\u6b65|\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728|"
        "\\u6211\\u8fd9\\u8fb9\\u5148|\\u6211\\u8fd9\\u8fb9\\u73b0\\u5728)?"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u8bb0\\u5f55|\\u8865\\u9f50|\\u8bb0\\u4e0a|\\u5199\\u5165|\\u5199\\u4e0a|"
        "\\u63d0\\u4ea4|\\u8865\\u5f55|\\u5bf9\\u9f50)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d|\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5e26\\u4f60|\\u518d\\u5e26\\u4f60|\\u7136\\u540e\\u5e26\\u4f60|"
        "\\u8fdb\\u4e0b\\u4e00\\u6b65|\\u505a\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u786e\\u8ba4|\\u8bfb\\u4e00\\u4e0b|\\u770b\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65|\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8981\\u6c42|\\u8bb0\\u5f55|\\u7136\\u540e|\\u518d\\u7ee7\\u7eed|\\u518d\\u5e26\\u4f60)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u786e\\u8ba4|\\u8bfb\\u4e00\\u4e0b|\\u770b\\u4e00\\u4e0b|\\u68b3\\u7406\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65|\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:"
        "\\u53ea\\u9700\\u8981\\u4f60\\u505a\\u4ec0\\u4e48(?:\\u3001|,|\\u548c)?\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u53ea\\u9700\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u53ea\\u9700\\u8981\\u4f60\\u505a\\u4ec0\\u4e48|"
        "\\u505a\\u4ec0\\u4e48(?:\\u3001|,|\\u548c)?\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u56de\\u62a5\\u4ec0\\u4e48"
        ")"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u8865\\u8bb0|\\u8bb0\\u4e00\\u4e0b|\\u8865\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u8f6e\\u7ed3\\u679c|\\u8fd9\\u4e00\\u8f6e|\\u5f53\\u524d\\u7ed3\\u679c|\\u5f53\\u524d\\u8fd9\\u4e00\\u8f6e)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u6838\\u5bf9|\\u786e\\u8ba4|\\u770b\\u4e00\\u773c|\\u770b\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u540e\\u9762\\u7684\\u7d27\\u63a5\\u6b65\\u9aa4|\\u540e\\u9762\\u7d27\\u63a5\\u6b65\\u9aa4|\\u540e\\u9762\\u6b65\\u9aa4|\\u7d27\\u63a5\\u6b65\\u9aa4|\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u770b\\u4e00\\u773c|\\u770b\\u4e00\\u4e0b|\\u786e\\u8ba4|\\u68b3\\u7406)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u6b65\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|\\u8fd9\\u4e00\\u6b65\\u8981\\u56de\\u62a5\\u4ec0\\u4e48|\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|\\u8981\\u56de\\u62a5\\u4ec0\\u4e48)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
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


CANONICAL_OPENING_RE = re.compile(
    r"今天我们做《[^》\n]{1,80}》。你准备好开始了吗[？?]"
)


def _collapse_duplicated_canonical_opening(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""

    opening_match = CANONICAL_OPENING_RE.match(normalized)
    if not opening_match:
        return normalized

    opening = opening_match.group(0)
    opening_without_q = re.sub(r"[？?]\s*$", "", opening)
    repeated_opening_re = re.compile(
        rf"^(?:{re.escape(opening)}\s*)+(?:{re.escape(opening_without_q)}\s*)?$"
    )
    if repeated_opening_re.fullmatch(normalized):
        return opening

    return normalized


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


def _is_full_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in BACKSTAGE_FULL_SENTENCE_PATTERNS)


def _is_structural_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in STRUCTURAL_BACKSTAGE_PATTERNS)


def _is_transition_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in TRANSITION_BACKSTAGE_PATTERNS)


def filter_spoken_backstage_text(text):
    """Drop backend workflow narration while preserving complete student-facing instructions."""
    if not text:
        return text

    normalized = _collapse_duplicated_canonical_opening(text)
    if not normalized:
        return ""

    # Hard opening rule: if the canonical opening sentence exists, keep only that sentence.
    opening_match = CANONICAL_OPENING_RE.search(normalized)
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
        if _is_full_backstage_sentence(s):
            return True
        if _is_structural_backstage_sentence(s):
            return True
        if _is_transition_backstage_sentence(s):
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
        return "".join(kept_sentences).strip()

    # Fallback: strip leading backstage clauses if possible; if the whole line is
    # backstage narration, return empty instead of replaying it.
    cleaned = _strip_backstage_leading_clauses(normalized).strip()
    if cleaned and cleaned != normalized:
        return filter_spoken_backstage_text(cleaned)

    rewritten = _rewrite_backstage_sentence(normalized).strip()
    if rewritten and rewritten != normalized:
        return filter_spoken_backstage_text(rewritten)

    if (
        _is_full_backstage_sentence(normalized)
        or _is_structural_backstage_sentence(normalized)
        or _is_transition_backstage_sentence(normalized)
        or _is_backstage_filler_sentence(normalized)
        or _looks_like_backstage_sentence(normalized)
    ):
        return ""

    return rewritten

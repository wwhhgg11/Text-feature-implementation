from collections import Counter
import math
from typing import Dict, List, Tuple, Iterable, Any
import torch
from ltp import LTP


# ======================
# 基础工具
# ======================
def _safe_log2(p: float) -> float:
    return math.log(p, 2) if p > 0 else 0.0

def counts_to_entropy(counter: Counter) -> float:
    """Shannon 熵（bit）。counter: {项: 频数}"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c/total) * _safe_log2(c/total) for c in counter.values() if c > 0)

def kurtosis_like(prob_list: List[float]) -> float:
    """
    语义噪音 SN 的峰度公式（⑧）:
    SN = n * sum((Pi - Pbar)^4) / ( sum((Pi - Pbar)^2)^2 )
    """
    n = len(prob_list)
    if n == 0:
        return 0.0
    mean = sum(prob_list) / n
    s2 = sum((p - mean) ** 2 for p in prob_list)
    if s2 == 0:
        return 0.0
    s4 = sum((p - mean) ** 4 for p in prob_list)
    return n * s4 / (s2 ** 2)

# ======================
# 解析 LTP 输出（适配多种 dep 形态）
# ======================
def extract_from_ltp_output(
    output: Any,
    use_lower: bool = True,
    exclude_punct_tokens: bool = True,
    punct_pos_tags: Tuple[str, ...] = ("wp",)  # LTP 常见：wp=标点
) -> Tuple[List[str], List[str], List[str]]:
    """
    从 LTP pipeline 输出中提取：
    - 词元：all_tokens：全部 token 列表（可选择排除标点）
    - 词性：all_pos：对应 POS 列表（与 tokens 对齐）
    - 依存关系标签：all_dep_labels：依存关系标签列表（可包含标点依存，后续函数里再决定是否剔除）

    兼容的 dep 形态：
    1) list[dict]: [{'head':[...], 'label':[...]}]  ← 你给的这种
    2) list[list[tuple]]: [[(head, label), ...], ...]
    3) list[list[dict]]: [[{'head':h, 'label':lab}, ...], ...]
    """
    all_tokens: List[str] = []
    all_pos: List[str] = []
    all_dep_labels: List[str] = []

    # ---- 先把 tokens/pos 拉平（可排除标点）
    for sent_idx in range(len(output.cws)):
        tokens = output.cws[sent_idx]
        pos_tags = output.pos[sent_idx]
        if use_lower:
            tokens = [t.lower() for t in tokens]

        if exclude_punct_tokens:
            for t, p in zip(tokens, pos_tags):
                if p not in punct_pos_tags and t.strip():
                    all_tokens.append(t)
                    all_pos.append(p)
        else:
            all_tokens.extend(tokens)
            all_pos.extend(pos_tags)

    # ---- 再解析依存标签（dep）
    # output.dep 的每个元素代表一条句子
    for dep_sent in output.dep:
        # 情况 A：你提供的这种： [{'head':[...], 'label':[...]}]
        if isinstance(dep_sent, list) and len(dep_sent) == 1 \
           and isinstance(dep_sent[0], dict) \
           and isinstance(dep_sent[0].get("label", None), list):
            labels = dep_sent[0]["label"]
            all_dep_labels.extend(labels)

        # 情况 B：list[tuple] 或 list[list/tuple]
        elif isinstance(dep_sent, list) and dep_sent and isinstance(dep_sent[0], (tuple, list, dict)):
            for d in dep_sent:
                if isinstance(d, (tuple, list)) and len(d) >= 2:
                    all_dep_labels.append(d[1])
                elif isinstance(d, dict) and "label" in d:
                    all_dep_labels.append(d["label"])

        # 情况 C：直接是 dict
        elif isinstance(dep_sent, dict) and isinstance(dep_sent.get("label", None), list):
            all_dep_labels.extend(dep_sent["label"])

        # 其他情况：忽略
        else:
            continue

    return all_tokens, all_pos, all_dep_labels

def get_token_and_noun_probs(
    all_tokens: List[str],
    all_pos: List[str],
    noun_prefix: Tuple[str, ...] = ("n",),
    exclude_punct_tokens: bool = True,
    punct_pos_tags: Tuple[str, ...] = ("wp",)
) -> Tuple[Counter, Counter, List[float], int]:
    """
    计算：
    - token_counter: 全词项频数（可选择已剔除标点）
    - noun_counter: 名词类型频数（按 POS 前缀判定）
    - noun_prob_list: 每个“名词类型”的概率 Pi（分母=总 token 数）
    - N_tokens: 总 token 数
    """
    # tokens/pos 已在 extract 中排过标点时，这里就不再重复排
    token_counter = Counter(all_tokens)
    N_tokens = sum(token_counter.values())

    noun_tokens = [
        tok for tok, pos in zip(all_tokens, all_pos)
        if any(pos.startswith(pref) for pref in noun_prefix)
    ]
    noun_counter = Counter(noun_tokens)
    noun_prob_list = [(c / N_tokens) for c in noun_counter.values()] if N_tokens > 0 else []
    return token_counter, noun_counter, noun_prob_list, N_tokens



from collections import Counter
import math
from typing import Dict, List, Tuple, Iterable, Any
import torch
from ltp import LTP
from utils import kurtosis_like, counts_to_entropy
import OpenHowNet
# 初始化 HowNet 字典
hownet_dict = OpenHowNet.HowNetDict()

# 指标函数（独立）
def metric_SR(noun_prob_list: List[float]) -> float:
    """语义丰富度 SR（⑥）：SR = - sum(Pi) over 名词类型"""
    return -sum(noun_prob_list) if noun_prob_list else 0.0

def metric_SC(noun_prob_list: List[float], N_tokens: int) -> float:
    """
    语义清晰度 SC（⑦，Lee 变体）:
    SC = (1/N) * sum( max(P) - Pi ) over 名词类型
    """
    if not noun_prob_list or N_tokens <= 0:
        return 0.0
    maxP = max(noun_prob_list)
    return sum(maxP - p for p in noun_prob_list) / N_tokens

def metric_SN(noun_prob_list: List[float]) -> float:
    """语义噪音 SN（⑧）：峰度公式"""
    return kurtosis_like(noun_prob_list)

def metric_lexical_entropy(token_counter: Counter) -> float:
    """词汇丰富度（Lexical Richness）：全词项分布的 Shannon 熵（公式④）"""
    return counts_to_entropy(token_counter)

def metric_syntactic_entropy(
    dep_labels: List[str],
    exclude_punc_deps: bool = False,
    punc_labels: Tuple[str, ...] = ("mPUNC", "WP", "PUNC", "mPunc", "mPunc".lower())
) -> Tuple[float, float]:
    """
    句法丰富度：依存关系标签的 Shannon 熵
    返回： (H, H_norm)
    - exclude_punc_deps=True 时剔除标点依存（如 mPUNC）
    """
    if exclude_punc_deps:
        labels = [lab for lab in dep_labels if lab not in punc_labels]
    else:
        labels = dep_labels[:]

    dep_counter = Counter(labels)
    H = counts_to_entropy(dep_counter)
    k = len(dep_counter)
    H_norm = H / math.log(k, 2) if k > 1 else 0.0
    return H, H_norm


def compute_SA(tokens, pos_tags, valid_pos_prefix=("n", "v", "a", "d")):
    """
    计算文本的语义精确度 (Semantic Accuracy, SA)

    参数：
    - tokens: List[str]，分词结果
    - pos_tags: List[str]，对应词性结果
    - valid_pos_prefix: Tuple[str]，表示哪些 POS 属于实词（名词n，动词v，形容词a，副词d）

    返回：
    - SA: float，语义精确度
    - details: dict，{word: sense_count}
    """
    sense_counts = []
    # details = {}

    for tok, pos in zip(tokens, pos_tags):
        # 判断是否属于实词
        if any(pos.startswith(pref) for pref in valid_pos_prefix):
            senses = hownet_dict.get_sense(tok)
            count = len(senses)
            sense_counts.append(count)
            #details[tok] = count

    if not sense_counts:
        return 0.0, {}

    # SA = 义项数总和 / 实词总数
    SA = sum(sense_counts) / len(sense_counts)
    return SA
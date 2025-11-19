from collections import Counter
import math
from typing import Dict, List, Tuple, Iterable, Any
import torch
from ltp import LTP
from utils import extract_from_ltp_output,get_token_and_noun_probs
from Metrics import metric_SR,metric_SC,metric_SN,metric_lexical_entropy,metric_syntactic_entropy,compute_SA
import pandas as pd
from types import SimpleNamespace
from tqdm import tqdm
# 汇总计算
def compute_all_metrics(
    output: Any,
    noun_prefix: Tuple[str, ...] = ("n",),        # 以这些前缀判定名词
    use_lower: bool = True,
    exclude_punct_tokens: bool = True,            # 词熵/名词概率是否剔除标点 token
    exclude_punc_deps: bool = False               # 句法熵是否剔除标点依存
) -> Dict[str, Any]:
    """
    一次性返回：SR, SC, SN, lexical_entropy, syntactic_entropy, syntactic_entropy_norm, 以及一些计数
    """
    all_tokens, all_pos, all_dep_labels = extract_from_ltp_output(
        output,
        use_lower=use_lower,
        exclude_punct_tokens=exclude_punct_tokens
    )

    token_counter, noun_counter, noun_prob_list, N_tokens = get_token_and_noun_probs(
        all_tokens, all_pos, noun_prefix=noun_prefix,
        exclude_punct_tokens=exclude_punct_tokens
    )

    SR = metric_SR(noun_prob_list)
    SC = metric_SC(noun_prob_list, N_tokens)
    SN = metric_SN(noun_prob_list)
    lexical_entropy = metric_lexical_entropy(token_counter)
    syn_H, syn_Hn = metric_syntactic_entropy(all_dep_labels, exclude_punc_deps=exclude_punc_deps)
    SA = compute_SA(all_tokens, all_pos)

    return {
        "SR": round(SR,2),
        "SC": round(SC,2),
        "SN": round(SN,2),
        "SA":round(SA,2),
        "lexical_entropy": round(lexical_entropy,2),
        "syntactic_entropy": round(syn_H,2),
        "syntactic_entropy_norm": round(syn_Hn,2),
        "counts": {
            "tokens": N_tokens,
            "noun_types": len(noun_counter),
            "dep_types": len(Counter(all_dep_labels))
        }
    }

# 模型文件
ltp = LTP(r"E:\A-word_complexity\Model\ltp-small")  # 默认加载 Small 模型
                        # 也可以传入模型的路径，ltp = LTP("/path/to/your/model")
                        # /path/to/your/model 应当存在 config.json 和其他模型文件
# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")
#
# #  分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
# output = ltp.pipeline(["春节把许多种情绪塞进短短几天，有想念，有温情，但也有回家之后，家里亲戚聚在一起闲谈时，暗中在职业、收入、房车等方面“比个高低。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
# # 使用字典格式作为返回结果
# results = compute_all_metrics(
#     output,
#     noun_prefix=("n","ns","nz"),   # 可按你的 POS 集合调整
#     use_lower=True,
#     exclude_punct_tokens=True,     # 词熵/名词概率里去掉标点
#     exclude_punc_deps=True         # 句法熵里去掉 mPUNC/标点依存
# )
# print(results)

def main():
    # ========= 路径参数 =========
    CSV_PATH = r"E:\A-word_complexity\Dataset\Weibo_Unemployment\Weibo_utils_clean_with_counts.csv"
    TEXT_COL = "text_clean"
    SAVE_PATH = CSV_PATH.replace(".csv", "_with_metrics.csv")

    # ========= 读入数据 =========
    df = pd.read_csv(CSV_PATH)
    texts = df[TEXT_COL].astype(str).fillna("").tolist()

    # 结果缓存
    SRs, SCs, SNs = [], [], []
    LEXs, SYNs, SYN_norms = [], [], []
    SAs = []   # 语义精确度

    BATCH = 64
    TASKS = ["cws", "pos", "dep"]

    total = len(texts)
    pbar = tqdm(total=total, desc="Processing", unit="sent")

    for start in range(0, total, BATCH):
        batch = texts[start:start + BATCH]

        with torch.inference_mode():
            out = ltp.pipeline(batch, tasks=TASKS)  # GPU 推理（你已 ltp.to("cuda")）

        for idx in range(len(out.cws)):
            one = SimpleNamespace(cws=[out.cws[idx]], pos=[out.pos[idx]], dep=[out.dep[idx]])

            metrics = compute_all_metrics(
                one,
                noun_prefix=("n", "ns", "nz"),
                use_lower=True,
                exclude_punct_tokens=True,
                exclude_punc_deps=True
            )
            SRs.append(metrics["SR"]);
            SCs.append(metrics["SC"]);
            SNs.append(metrics["SN"])
            LEXs.append(metrics["lexical_entropy"]);
            SYNs.append(metrics["syntactic_entropy"])
            SYN_norms.append(metrics["syntactic_entropy_norm"])

            # —— SA：句内去重 + 兼容返回形式（float 或 (sa, details)）——
            seen = set()
            tok_u, pos_u = [], []
            for t, p in zip(out.cws[idx], out.pos[idx]):
                if (t, p) not in seen and (
                        p.startswith("n") or p.startswith("v") or p.startswith("a") or p.startswith("d")):
                    seen.add((t, p));
                    tok_u.append(t);
                    pos_u.append(p)

            res = compute_SA(tok_u, pos_u, valid_pos_prefix=("n", "v", "a", "d"))
            SA_val = res[0] if isinstance(res, (list, tuple)) else res
            SAs.append(SA_val)

            pbar.update(1)

    pbar.close()

    # ========= 写回 CSV =========
    df["SR"] = SRs
    df["SC"] = SCs
    df["SN"] = SNs
    df["LexicalEntropy"] = LEXs
    df["SyntacticEntropy"] = SYNs
    df["SyntacticEntropy_Norm"] = SYN_norms
    df["SA"] = SAs

    df.to_csv(SAVE_PATH, index=False)
    print(f"Done! Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()

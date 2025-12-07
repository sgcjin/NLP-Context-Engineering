import re
from typing import List, Set
import jieba
import stopwordsiso as stopwordsiso
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHINESE_STOPWORDS: Set[str] = set(stopwordsiso.stopwords("zh"))

# -------------------------------------------------------
# Tokenization + keyword extraction
# -------------------------------------------------------

def chinese_tokens(text: str) -> List[str]:

    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    segments = text.split()

    tokens: List[str] = []

    for seg in segments:
        if not seg:
            continue

        if re.search(r"[\u4e00-\u9fa5]", seg):
            seg_tokens = [t.strip() for t in jieba.lcut(seg) if t.strip()]
            tokens.extend(seg_tokens)
        else:
            tokens.append(seg.lower())

    return tokens

def extract_keywords(text: str) -> Set[str]:

    tokens = chinese_tokens(text)
    keywords: Set[str] = set()

    for t in tokens:
        if len(t) < 2 or t in CHINESE_STOPWORDS:
            continue
        keywords.add(t)

    return keywords

# -------------------------------------------------------
# Keyword-level F1
# -------------------------------------------------------

def keyword_f1_single(pred: str, gold: str) -> float:

    pred_kw = extract_keywords(pred)
    gold_kw = extract_keywords(gold)

    if not pred_kw or not gold_kw:
        return 0.0

    inter = pred_kw & gold_kw
    if not inter:
        return 0.0

    precision = len(inter) / len(pred_kw)
    recall = len(inter) / len(gold_kw)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def keyword_f1(pred: str, gold_list: List[str]) -> float:

    if not gold_list or not extract_keywords(pred):
        return 0.0
    best_f1 = 0.0
    for gold in gold_list:
        f1 = keyword_f1_single(pred, gold)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

# -------------------------------------------------------
# Reward
# -------------------------------------------------------

def compute_rewards_for_summaries(

    pred_answers: List[str],
    gold_answers_list: List[List[str]],
    summary_lengths: torch.Tensor | List[int],
    max_summary_len: int,
    length_penalty_lambda: float,
) -> torch.Tensor:

    assert max_summary_len > 0, "max_summary_len must be positive"

    if isinstance(summary_lengths, torch.Tensor):
        summary_lengths_list = summary_lengths.tolist()
    else:
        summary_lengths_list = summary_lengths

    rewards = []
    for pred, gold_list, L in zip(pred_answers, gold_answers_list, summary_lengths_list):
        qa_score = keyword_f1(pred, gold_list)

        length_ratio = min(L / max_summary_len, 1.0)

        penalty = length_penalty_lambda * length_ratio
        R = qa_score - penalty
        rewards.append(R)

    return torch.tensor(rewards, dtype=torch.float32, device=DEVICE)


# -------------------------------------------------------
# REINFORCE loss
# -------------------------------------------------------

def reinforce_loss_for_summaries(
    summary_log_probs: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:

    assert summary_log_probs.shape == rewards.shape, f"log_probs shape {summary_log_probs.shape} != rewards shape {rewards.shape}"

    baseline = rewards.mean()
    advantages = rewards - baseline
    advantages_detached = advantages.detach()

    loss = - (advantages_detached * summary_log_probs).mean()
    return loss
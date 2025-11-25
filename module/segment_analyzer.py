#!/usr/bin/env python
"""
Multilingual segment analyzer (English / Japanese / Korean)

- 텍스트 조각이 완전한 문장인지 평가하고, 끊어 읽었을 때의 자연스러움도 함께 계산합니다.
- 현재는 영어(en), 일본어(ja), 한국어(ko)를 지원합니다.
 - 한국어는 spaCy+Stanza(ko) 파이프라인을 사용하고,
   실패 시 간단한 규칙 기반 분석으로 대체합니다.

필요:
  pip install spacy
  python -m spacy download en_core_web_sm
  python -m spacy download ja_core_news_sm

  한국어:
    pip install stanza spacy-stanza
    python -m stanza.download ko
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, Iterable, List, Set, Tuple

import spacy
try:
    import spacy_stanza
except ImportError:  # spacy-stanza가 없을 수도 있으니 옵션 처리
    spacy_stanza = None

# PyTorch 2.6 weights_only 기본값으로 인한 stanza 로딩 실패 방지용 안전 등록
try:
    import numpy as _np
    from torch.serialization import add_safe_globals

    add_safe_globals([_np.core.multiarray._reconstruct])
except Exception:
    pass

logger = logging.getLogger(__name__)


def _load_stanza_ko_pipeline():
    """
    spaCy-Stanza ko 파이프라인을 로드합니다.
    - PyTorch 2.6부터 weights_only 기본값이 True로 바뀌어 stanza 로딩이 막히므로
      내부적으로 torch.load의 기본 weights_only를 False로 강제합니다.
    """
    if spacy_stanza is None:
        return None

    torch_module = None
    original_torch_load = None
    try:
        import torch as _torch

        torch_module = _torch
        original_torch_load = _torch.load

        def _patched_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        _torch.load = _patched_torch_load
    except Exception:
        # torch가 없거나 패치에 실패하면 그대로 진행
        pass

    try:
        logger.info("spaCy-Stanza 한국어 파이프라인을 사용합니다.")
        return spacy_stanza.load_pipeline("ko")
    finally:
        if torch_module is not None and original_torch_load is not None:
            torch_module.load = original_torch_load


@dataclass(frozen=True)
class LanguageConfig:
    model_name: str
    sent_end_punct: Set[str]
    bad_end_pos: Set[str]
    bad_start_pos: Set[str]
    short_ok_sentences: Set[str]
    bad_end_words: Set[str]
    case_sensitive: bool = False
    blank_fallbacks: Tuple[str, ...] = ()


LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        model_name="en_core_web_sm",
        sent_end_punct={".", "!", "?"},
        bad_end_pos={"ADP", "DET", "PART", "SCONJ"},
        bad_start_pos={"CCONJ", "SCONJ"},
        short_ok_sentences={
            "yes",
            "no",
            "okay",
            "ok",
            "thanks",
            "thank you",
            "sure",
        },
        bad_end_words={"to", "of", "in", "at", "for", "on", "with"},
        blank_fallbacks=("en",),
    ),
    "ja": LanguageConfig(
        model_name="ja_core_news_sm",
        sent_end_punct={"。", "！", "？", "!", "?"},
        bad_end_pos={"ADP", "SCONJ", "PART"},
        bad_start_pos={"CCONJ", "SCONJ", "ADV"},
        short_ok_sentences={"はい", "いいえ", "了解", "了解です", "ありがとう", "ありがとうございます", "どうも", "うん"},
        bad_end_words={"は", "が", "を", "に", "へ", "で", "と", "から", "まで", "より", "や", "の", "ね", "よ", "か", "も", "って"},
        case_sensitive=True,
        blank_fallbacks=("ja", "xx"),
    ),
    "ko": LanguageConfig(
        model_name="ko_core_news_sm",
        sent_end_punct={".", "!", "?", "！", "？"},
        bad_end_pos={"ADP", "SCONJ", "PART"},
        bad_start_pos={"CCONJ", "SCONJ", "ADV"},
        short_ok_sentences={"네", "예", "아니요", "응", "웅", "그래", "좋아", "고마워", "감사합니다", "괜찮아요", "괜찮아"},
        bad_end_words={"은", "는", "이", "가", "을", "를", "에", "에서", "께", "한테", "에게", "까지", "부터", "으로", "로", "와", "과", "랑", "하고", "도", "만", "같이", "처럼", "보다", "조차", "마저", "이나", "나", "요", "죠", "지"},
        case_sensitive=True,
        blank_fallbacks=("ko", "xx"),
    ),
}

DEFAULT_LANGUAGE = "en"


@lru_cache(maxsize=None)
def _load_model(language: str):
    """spaCy 모델을 lazy하게 로드합니다."""
    config = LANGUAGE_CONFIGS[language]

    # 한국어는 spacy-stanza 파이프라인만 사용
    if language == "ko":
        if spacy_stanza is not None:
            try:
                loaded = _load_stanza_ko_pipeline()
                if loaded is not None:
                    return loaded
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "spaCy-Stanza 한국어 파이프라인 로드 실패: %s",
                    exc,
                )
        else:
            logger.warning("spaCy-Stanza가 설치되어 있지 않아 한국어 모델을 사용할 수 없습니다.")

        logger.warning(
            "한국어에 대해 blank spaCy 모델로 대체합니다.",
        )
        for blank_code in config.blank_fallbacks:
            try:
                return spacy.blank(blank_code)
            except Exception:
                continue
        raise RuntimeError("언어 %s에 대한 NLP 파이프라인 로딩에 실패했습니다." % language)

    try:
        return spacy.load(config.model_name)
    except Exception as exc:  # pragma: no cover - 방어적 코드
        logger.warning(
            "spaCy 모델 로드 실패(%s): %s.",
            language,
            exc,
        )

    if language == "ko" and spacy_stanza is not None:
        try:
            loaded = _load_stanza_ko_pipeline()
            if loaded is not None:
                return loaded
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "spaCy-Stanza 한국어 파이프라인 로드 실패: %s",
                exc,
            )

    logger.warning(
        "언어 %s 에 대해 blank spaCy 모델로 대체합니다.",
        language,
    )
    for blank_code in config.blank_fallbacks:
        try:
            return spacy.blank(blank_code)
        except Exception:
            continue
    raise RuntimeError("언어 %s에 대한 NLP 파이프라인 로딩에 실패했습니다." % language)


def _get_nlp(language: str):
    return _load_model(language)


def _normalize_language(language: str) -> str:
    if not language:
        return DEFAULT_LANGUAGE
    normalized = language.lower()
    if normalized not in LANGUAGE_CONFIGS:
        logger.warning("지원하지 않는 형태소 분석 언어(%s)가 전달되어 영어로 대체합니다.", language)
        return DEFAULT_LANGUAGE
    return normalized


def _contains_korean(text: str) -> bool:
    """간단히 한글 여부를 확인."""
    for char in text:
        if "가" <= char <= "힣":
            return True
    return False


def _ends_with_particle(text: str, particles: Set[str]) -> bool:
    """조사/어미 같은 짧은 토큰으로 끝나는지 확인."""
    stripped = text.strip()
    if not stripped:
        return False
    for particle in particles:
        if stripped.endswith(particle):
            return True
    return False


def _looks_like_korean_verb(token_text: str) -> bool:
    """동사/형용사 활용 어미로 추정되는지 간단히 검사."""
    if not token_text or not _contains_korean(token_text):
        return False
    normalized = token_text.strip()
    verb_endings = (
        "다",
        "요",
        "니다",
        "했다",
        "된다",
        "됐다",
        "였다",
        "한다",
        "했다가",
        "한다가",
        "할게",
        "할게요",
        "할께",
        "할께요",
        "해라",
        "하세요",
        "하십시오",
        "합시다",
        "하자",
        "해요",
        "해",
        "줘",
        "줘요",
        "줘라",
        "봐",
        "봐요",
        "봐라",
        "지",
        "지요",
        "지요?",
        "겠어",
        "겠습니다",
    )
    return normalized.endswith(verb_endings)


@dataclass
class SegmentAnalysis:
    text: str
    language: str
    tokens: List[str]
    is_complete_sentence: bool
    completeness_score: float   # 0.0 ~ 1.0
    break_naturalness: float    # 0.0 ~ 1.0 (높을수록 '이대로 끊어도 덜 어색')
    ok_as_segment: bool         # break_naturalness 기준으로 적당한지 여부
    reasons: List[str]          # 점수에 영향을 준 이유들(디버깅용)


def _has_finite_verb(doc: Iterable, language: str) -> bool:
    """시제/인칭이 있는 동사가 있는지 (대충 '문장 같다'의 핵심 조건)."""
    if language == "ja":
        return any(token.pos_ in {"VERB", "AUX"} for token in doc)
    if language == "ko":
        return any(_looks_like_korean_verb(token.text) for token in doc)

    for token in doc:
        if token.pos_ in {"VERB", "AUX"}:
            verb_forms = token.morph.get("VerbForm")
            # VerbForm 정보가 없거나 Fin 포함이면 유한동사로 간주
            if not verb_forms or "Fin" in verb_forms:
                return True
    return False


def _has_subject(doc: Iterable, language: str) -> bool:
    """주어가 있는지 확인."""
    if language == "ja":
        for token in doc:
            if token.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                return True
            if token.pos_ in {"NOUN", "PROPN", "PRON"}:
                for child in token.children:
                    if child.pos_ in {"ADP", "PART"} and child.text in {"は", "が"}:
                        return True
        return False
    if language == "ko":
        for token in doc:
            token_text = token.text.strip()
            if not token_text or not _contains_korean(token_text):
                continue
            if _ends_with_particle(token_text, {"은", "는", "이", "가", "께서", "께"}):
                return True
        return False

    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"}:
            return True
    return False


def _looks_imperative_en(doc) -> bool:
    """
    명령문 형태인지 대충 체크:
    - 주어(nsubj)가 없고
    - 첫 토큰이 동사(VB)일 때
    """
    if not doc:
        return False

    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass", "csubj"}:
            return False

    first = doc[0]
    if first.pos_ == "VERB" and first.tag_ == "VB":
        return True

    return False


def _looks_imperative_ja(doc) -> bool:
    if not doc:
        return False

    last = doc[-1]
    if last.pos_ in {"VERB", "AUX"}:
        verb_forms = last.morph.get("VerbForm")
        if verb_forms and "Imp" in verb_forms:
            return True
    last_text = last.lemma_ or last.text
    return last_text.endswith(("て", "で", "なさい", "ください", "ろ", "よ"))


def _looks_imperative_ko(doc) -> bool:
    if not doc:
        return False

    last_text = doc[-1].text.strip()
    if not last_text or not _contains_korean(last_text):
        return False

    imperative_endings = (
        "해라",
        "해봐",
        "해봐요",
        "해봐라",
        "해보세요",
        "해줘",
        "해줘요",
        "해줘라",
        "하라",
        "하세요",
        "하십시오",
        "합시다",
        "하자",
        "가라",
        "와라",
        "앉아",
        "앉아요",
        "봐라",
        "봐요",
        "들어라",
        "들어요",
        "하지마",
        "하지 마",
        "하지 마라",
        "하지 마세요",
        "말아",
        "말아요",
        "말아요?",
    )
    return last_text.endswith(imperative_endings)


def _looks_imperative(doc, language: str) -> bool:
    if language == "ko":
        return _looks_imperative_ko(doc)
    if language == "ja":
        return _looks_imperative_ja(doc)
    return _looks_imperative_en(doc)


def _has_unmatched_quotes_or_parens(text: str) -> bool:
    """따옴표/괄호가 짝이 안 맞는지 간단히 체크."""
    # 큰따옴표 짝
    double_quotes = text.count('"')
    if double_quotes % 2 == 1:
        return True

    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return True
            stack.pop()

    return bool(stack)


def _normalize_text(text: str, config: LanguageConfig) -> str:
    return text if config.case_sensitive else text.lower()


def analyze_segment(text: str, language: str = DEFAULT_LANGUAGE) -> SegmentAnalysis:
    """주어진 텍스트 조각에 대해 분석을 수행합니다."""
    stripped = text.strip()
    normalized_language = _normalize_language(language)
    config = LANGUAGE_CONFIGS[normalized_language]
    doc = _get_nlp(normalized_language)(stripped)
    tokens = [t.text for t in doc if not t.is_space]
    content_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    if not content_tokens:
        return SegmentAnalysis(
            text=text,
            language=normalized_language,
            tokens=[],
            is_complete_sentence=False,
            completeness_score=0.0,
            break_naturalness=0.0,
            ok_as_segment=False,
            reasons=["empty"],
        )
    normalized_text = _normalize_text(stripped, config)

    reasons: List[str] = []

    length = len(content_tokens)
    last_token = content_tokens[-1]
    first_token = content_tokens[0]

    has_finite_verb = _has_finite_verb(doc, normalized_language)
    has_subject = _has_subject(doc, normalized_language)
    looks_imperative = _looks_imperative(doc, normalized_language)

    # ---------- 1) 문장 완전성 점수 ----------
    score = 0.0

    if has_finite_verb:
        score += 0.4
        reasons.append("finite_verb")

    if has_subject or looks_imperative:
        score += 0.3
        reasons.append("subject_or_imperative")

    normalized_last = _normalize_text(last_token.text, config)
    if length >= 4:
        score += 0.1

    # 구두점으로 끝날 때 완전성 가중치 추가 (언어별 종결부호 고려)
    if normalized_language in ("en", "ja", "ko"):
        for token in reversed(doc):
            if token.is_space:
                continue
            if token.is_punct:
                token_text = token.text.strip()
                if token_text in {".", "!", "?", "。", "！", "？"}:
                    score += 0.1
                    reasons.append("punct_bonus_end")
                elif token_text in {",", "，", "、"}:
                    score += 0.05
                    reasons.append("comma_bonus_end")
            break

    if any(t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"} for t in doc):
        score += 0.1
        reasons.append("verbal_root")

    # 아주 짧아도 자연스러운 상용 표현
    if length <= 3 and normalized_text in config.short_ok_sentences:
        score = max(score, 0.8)
        reasons.append("short_but_common")

    completeness_score = max(0.0, min(1.0, score))
    is_complete = completeness_score >= 0.7

    # ---------- 2) 조각으로 끊을 때 자연스러운지 ----------
    awkward = 0.4  # 기본은 '그럭저럭'

    if not is_complete:
        awkward += 0.1  # 완전한 문장이 아니면 조금 감점

    # 끝이 전치사/관사/접속사 등이면 어색
    if last_token.pos_ in config.bad_end_pos:
        awkward += 0.3
        reasons.append(f"bad_end_pos:{last_token.pos_}")

    if normalized_last in config.bad_end_words:
        awkward += 0.2
        reasons.append(f"bad_end_word:{normalized_last}")
    elif normalized_language == "ko" and _ends_with_particle(last_token.text, config.bad_end_words):
        awkward += 0.2
        reasons.append("bad_end_particle")

    # 시작이 접속사(And, But, Because...)이면 어색한 조각일 가능성
    if first_token.pos_ in config.bad_start_pos:
        awkward += 0.2
        reasons.append(f"bad_start_pos:{first_token.pos_}")

    # 너무 짧은 조각은 (예외 리스트 외에는) 어색하다고 봄
    if length <= 2 and normalized_text not in config.short_ok_sentences:
        awkward += 0.2
        reasons.append("too_short")

    # 따옴표/괄호 짝이 안 맞으면 어색
    if _has_unmatched_quotes_or_parens(stripped):
        awkward += 0.2
        reasons.append("unmatched_quotes_or_parens")

    awkward = max(0.0, min(1.0, awkward))
    break_naturalness = 1.0 - awkward  # 높을수록 자연스러운 끊김

    # 임계값은 필요에 따라 조정 가능
    ok_as_segment = break_naturalness >= 0.5

    return SegmentAnalysis(
        text=text,
        language=normalized_language,
        tokens=tokens,
        is_complete_sentence=is_complete,
        completeness_score=round(completeness_score, 3),
        break_naturalness=round(break_naturalness, 3),
        ok_as_segment=ok_as_segment,
        reasons=reasons,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze segments for sentence completeness and break naturalness."
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="분석할 텍스트 (여러 단어를 그대로 붙여서 하나의 세그먼트로 취급합니다).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="사람이 읽기 좋은 텍스트 대신 JSON으로 출력합니다.",
    )
    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGE_CONFIGS.keys()),
        default=DEFAULT_LANGUAGE,
        help="분석에 사용할 언어 코드 (기본값: en).",
    )
    args = parser.parse_args()

    # 공백으로 join 해서 하나의 세그먼트로 처리
    joined_text = " ".join(args.text)

    analysis = analyze_segment(joined_text, language=args.language)

    if args.json:
        print(json.dumps(asdict(analysis), ensure_ascii=False, indent=2))
    else:
        print(f"TEXT: {analysis.text}")
        print(f"Language: {analysis.language}")
        print(f"Tokens: {analysis.tokens}")
        print(
            f"Complete sentence: {analysis.is_complete_sentence} "
            f"(score={analysis.completeness_score})"
        )
        print(
            f"Break naturalness: {analysis.break_naturalness} "
            f"(ok_as_segment={analysis.ok_as_segment})"
        )
        print(f"Reasons: {', '.join(analysis.reasons)}")


if __name__ == "__main__":
    main()

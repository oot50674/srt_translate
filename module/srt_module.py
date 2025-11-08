import logging
import os
import re
from typing import List, Dict, Any, TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Subtitle(TypedDict):
    index: int
    start: str
    end: str
    text: str

SRT_PATTERN = re.compile(
    r"(\d+)\s+([0-9:,]+)\s-->\s([0-9:,]+)\s+(.*?)\s*(?=\n\d+\s|\Z)",
    re.DOTALL,
)


def _parse_srt_content(content: str) -> List[Subtitle]:
    """주어진 SRT 형식 문자열을 파싱하여 자막 정보를 반환합니다."""
    subtitles: List[Subtitle] = []
    for match in SRT_PATTERN.finditer(content):
        index = int(match.group(1))
        start = match.group(2).strip()
        end = match.group(3).strip()
        text = match.group(4).replace("\n", " ").strip()
        subtitles.append({
            "index": index,
            "start": start,
            "end": end,
            "text": text,
        })
    logger.info("Parsed %d subtitles", len(subtitles))
    return subtitles


def read_srt(file_path: str) -> List[Subtitle]:
    """SRT 파일을 읽어 자막 정보를 리스트로 반환합니다.

    Args:
        file_path (str): 읽을 SRT 파일 경로.

    Returns:
        list[dict]: 각 자막 블록을 나타내는 딕셔너리 리스트.
    """
    logger.info("Loading SRT file: %s", file_path)

    # 안전한 인코딩 처리: 여러 인코딩을 시도해 한글 깨짐을 방지합니다.
    # 우선 바이너리로 읽은 뒤 여러 디코딩을 시도해 성공한 결과를 사용합니다.
    tried_encodings = []
    with open(file_path, "rb") as bf:
        raw = bf.read()

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "iso-8859-1"):
        try:
            content = raw.decode(enc)
            logger.info("Decoded SRT using encoding=%s for file %s", enc, file_path)
            return _parse_srt_content(content)
        except Exception:
            tried_encodings.append(enc)
            continue

    # 모든 디코딩 실패시 마지막으로 latin-1로 강제 디코드하고 파싱 시도
    logger.warning(
        "모든 시도한 인코딩(%s)에서 디코딩 실패: %s — latin-1로 강제 디코딩 시도",
        tried_encodings,
        file_path,
    )
    content = raw.decode("iso-8859-1", errors="replace")
    return _parse_srt_content(content)


def parse_srt_text(content: str) -> List[Subtitle]:
    """SRT 형식의 문자열을 파싱하여 자막 리스트를 반환합니다."""
    return _parse_srt_content(content)


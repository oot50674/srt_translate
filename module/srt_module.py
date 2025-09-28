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

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    return _parse_srt_content(content)


def parse_srt_text(content: str) -> List[Subtitle]:
    """SRT 형식의 문자열을 파싱하여 자막 리스트를 반환합니다."""
    return _parse_srt_content(content)


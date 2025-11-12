"""자막 보정 싱크 모듈 - Silero VAD를 활용한 자막 타이밍 보정"""
from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from module.silero_vad import SileroVAD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """자막 싱크 보정 설정"""
    # 청크 모드
    chunk_mode: str = "grouped"  # "grouped" or "individual"

    # 청크 묶기 설정
    gap_threshold_ms: int = 200

    # 경계 탐색 범위
    lookback_start_ms: int = 800
    lookahead_start_ms: int = 400
    lookback_end_ms: int = 400
    lookahead_end_ms: int = 800

    # 이동 제약
    max_left_shift_ms: int = 500
    max_left_expand_ms: int = 800
    max_right_shift_ms: int = 500
    max_right_expand_ms: int = 800

    # 패딩
    pad_ms: int = 80

    # 최소 엔트리 길이
    min_entry_duration_ms: int = 300

    # 신뢰도 임계값
    threshold_score: float = 0.3

    # VAD 설정
    vad_threshold: float = 0.55
    vad_min_speech_duration_ms: int = 200
    vad_min_silence_duration_ms: int = 250
    vad_speech_pad_ms: int = 80

    # 내부 엔트리 처리 모드
    entry_mode: str = "edge-only"  # "edge-only" or "proportional"

    # 겹침 보정 설정
    fix_overlaps: bool = True  # 겹침 자동 보정 활성화
    min_gap_ms: int = 10  # 엔트리 간 최소 간격 (밀리초)


@dataclass
class SubtitleEntry:
    """자막 엔트리"""
    index: int
    start_ms: float
    end_ms: float
    text: str

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Chunk:
    """자막 청크 (연속된 엔트리 묶음)"""
    entries: List[SubtitleEntry]
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def time_to_ms(time_str: str) -> float:
    """SRT 시간 형식을 밀리초로 변환

    Args:
        time_str: "HH:MM:SS,mmm" 형식의 시간 문자열

    Returns:
        밀리초 단위 시간
    """
    try:
        time_part, ms_part = time_str.strip().split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return (h * 3600 + m * 60 + s) * 1000 + ms
    except Exception as e:
        logger.error(f"시간 변환 실패: {time_str}, 에러: {e}")
        return 0.0


def ms_to_time(ms: float) -> str:
    """밀리초를 SRT 시간 형식으로 변환

    Args:
        ms: 밀리초 단위 시간

    Returns:
        "HH:MM:SS,mmm" 형식의 시간 문자열
    """
    ms = max(0, ms)  # 음수 방지
    total_seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt_entries(subtitles: List[Dict]) -> List[SubtitleEntry]:
    """SRT 자막 리스트를 SubtitleEntry 리스트로 변환

    Args:
        subtitles: srt_module에서 파싱된 자막 리스트

    Returns:
        SubtitleEntry 리스트
    """
    entries = []
    for sub in subtitles:
        try:
            entry = SubtitleEntry(
                index=sub['index'],
                start_ms=time_to_ms(sub['start']),
                end_ms=time_to_ms(sub['end']),
                text=sub['text']
            )
            entries.append(entry)
        except Exception as e:
            logger.warning(f"엔트리 변환 실패: {sub}, 에러: {e}")

    return entries


def build_chunks(entries: List[SubtitleEntry], chunk_mode: str, gap_threshold_ms: int) -> List[Chunk]:
    """자막 엔트리들을 청크로 묶기

    Args:
        entries: 자막 엔트리 리스트
        chunk_mode: 청크 모드 ("grouped" or "individual")
        gap_threshold_ms: 청크를 나누는 간격 임계값 (밀리초, grouped 모드에서만 사용)

    Returns:
        청크 리스트
    """
    if not entries:
        return []

    chunks = []

    if chunk_mode == "individual":
        # 개별 모드: 각 엔트리를 독립적인 청크로 처리
        for entry in entries:
            chunk = Chunk(
                entries=[entry],
                start_ms=entry.start_ms,
                end_ms=entry.end_ms
            )
            chunks.append(chunk)
        logger.info(f"개별 모드: {len(entries)}개 엔트리를 {len(chunks)}개 청크로 분리")
        return chunks

    # grouped 모드: 이어진 자막들을 하나의 청크로 묶기
    current_chunk_entries = [entries[0]]

    for i in range(1, len(entries)):
        prev_entry = entries[i - 1]
        curr_entry = entries[i]

        # 이전 엔트리 끝과 현재 엔트리 시작 사이의 간격
        gap = curr_entry.start_ms - prev_entry.end_ms

        if gap <= gap_threshold_ms:
            # 같은 청크에 포함
            current_chunk_entries.append(curr_entry)
        else:
            # 새로운 청크 시작
            chunk = Chunk(
                entries=current_chunk_entries,
                start_ms=current_chunk_entries[0].start_ms,
                end_ms=current_chunk_entries[-1].end_ms
            )
            chunks.append(chunk)
            current_chunk_entries = [curr_entry]

    # 마지막 청크 추가
    if current_chunk_entries:
        chunk = Chunk(
            entries=current_chunk_entries,
            start_ms=current_chunk_entries[0].start_ms,
            end_ms=current_chunk_entries[-1].end_ms
        )
        chunks.append(chunk)

    logger.info(f"그룹 모드: 총 {len(entries)}개 엔트리를 {len(chunks)}개 청크로 묶음")
    return chunks


def merge_close_segments(
    segments: List[Dict],
    merge_threshold_ms: int = 150
) -> List[Dict]:
    """인접한 VAD 세그먼트들을 병합

    Args:
        segments: VAD 세그먼트 리스트 [{'start': s, 'end': e}, ...]
        merge_threshold_ms: 병합할 간격 임계값

    Returns:
        병합된 세그먼트 리스트
    """
    if not segments:
        return []

    # 시간 기준 정렬
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    merged = [sorted_segs[0].copy()]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        gap_ms = (seg['start'] - last['end']) * 1000

        if gap_ms <= merge_threshold_ms:
            # 병합
            last['end'] = max(last['end'], seg['end'])
        else:
            merged.append(seg.copy())

    return merged


def filter_short_segments(
    segments: List[Dict],
    min_duration_ms: int
) -> List[Dict]:
    """짧은 세그먼트 제거

    Args:
        segments: VAD 세그먼트 리스트
        min_duration_ms: 최소 세그먼트 길이

    Returns:
        필터링된 세그먼트 리스트
    """
    filtered = []
    for seg in segments:
        duration_ms = (seg['end'] - seg['start']) * 1000
        if duration_ms >= min_duration_ms:
            filtered.append(seg)

    return filtered


def find_best_boundary(
    boundary_ms: float,
    segments: List[Dict],
    lookback_ms: int,
    lookahead_ms: int,
    pad_ms: int,
    is_start: bool,
    config: SyncConfig
) -> Tuple[Optional[float], float]:
    """경계에 가장 적합한 VAD 지점 찾기

    Args:
        boundary_ms: 원본 경계 시간 (밀리초)
        segments: VAD 세그먼트 리스트 (초 단위)
        lookback_ms: 뒤로 탐색 범위
        lookahead_ms: 앞으로 탐색 범위
        pad_ms: 패딩
        is_start: True면 시작 경계, False면 끝 경계
        config: 싱크 설정

    Returns:
        (새로운 경계 시간, 신뢰도 점수) 튜플
    """
    if not segments:
        return None, 0.0

    # 탐색 구간 설정
    search_start_ms = boundary_ms - lookback_ms
    search_end_ms = boundary_ms + lookahead_ms

    # 탐색 구간과 겹치는 세그먼트 찾기
    candidates = []
    for seg in segments:
        seg_start_ms = seg['start'] * 1000
        seg_end_ms = seg['end'] * 1000

        # 세그먼트가 탐색 구간과 겹치는지 확인
        if seg_end_ms >= search_start_ms and seg_start_ms <= search_end_ms:
            candidates.append(seg)

    if not candidates:
        return None, 0.0

    # 후보 평가
    best_candidate = None
    best_score = -float('inf')

    for seg in candidates:
        seg_start_ms = seg['start'] * 1000
        seg_end_ms = seg['end'] * 1000
        seg_duration_ms = seg_end_ms - seg_start_ms

        if is_start:
            # 시작 경계: 세그먼트 시작점 기준
            target_ms = seg_start_ms
            distance = abs(target_ms - boundary_ms)
        else:
            # 끝 경계: 세그먼트 끝점 기준
            target_ms = seg_end_ms
            distance = abs(target_ms - boundary_ms)

        # 점수 계산: 거리가 가까울수록, 세그먼트가 길수록 높은 점수
        distance_score = 1.0 / (1.0 + distance / 100.0)  # 100ms당 점수 감소
        duration_score = min(seg_duration_ms / 1000.0, 1.0)  # 1초 이상이면 만점

        score = distance_score * 0.7 + duration_score * 0.3

        if score > best_score:
            best_score = score
            best_candidate = (target_ms, seg_duration_ms)

    if best_candidate is None:
        return None, 0.0

    target_ms, seg_duration_ms = best_candidate

    # 패딩 적용
    if is_start:
        new_boundary = target_ms - pad_ms
    else:
        new_boundary = target_ms + pad_ms

    return new_boundary, best_score


def snap_chunk_boundaries(
    chunk: Chunk,
    segments: List[Dict],
    config: SyncConfig,
    prev_chunk: Optional[Chunk],
    next_chunk: Optional[Chunk]
) -> Tuple[float, float]:
    """청크 경계를 VAD 세그먼트에 스냅

    Args:
        chunk: 보정할 청크
        segments: VAD 세그먼트 리스트
        config: 싱크 설정
        prev_chunk: 이전 청크 (없으면 None)
        next_chunk: 다음 청크 (없으면 None)

    Returns:
        (새로운 시작 시간, 새로운 끝 시간) 튜플 (밀리초)
    """
    new_start = chunk.start_ms
    new_end = chunk.end_ms

    # 시작 경계 스냅
    start_boundary, start_score = find_best_boundary(
        chunk.start_ms,
        segments,
        config.lookback_start_ms,
        config.lookahead_start_ms,
        config.pad_ms,
        is_start=True,
        config=config
    )

    if start_boundary is not None and start_score >= config.threshold_score:
        # 이동량 제약 확인
        delta = chunk.start_ms - start_boundary

        if delta >= 0:  # 왼쪽으로 이동 (앞당김)
            if delta <= config.max_left_shift_ms:
                new_start = start_boundary
            else:
                new_start = chunk.start_ms - config.max_left_shift_ms
        else:  # 오른쪽으로 이동 (뒤로 미룸)
            if abs(delta) <= config.max_left_expand_ms:
                new_start = start_boundary
            else:
                new_start = chunk.start_ms + config.max_left_expand_ms

        # 이전 청크와 겹치지 않도록
        if prev_chunk is not None:
            new_start = max(new_start, prev_chunk.end_ms)

    # 끝 경계 스냅
    end_boundary, end_score = find_best_boundary(
        chunk.end_ms,
        segments,
        config.lookback_end_ms,
        config.lookahead_end_ms,
        config.pad_ms,
        is_start=False,
        config=config
    )

    if end_boundary is not None and end_score >= config.threshold_score:
        # 이동량 제약 확인
        delta = end_boundary - chunk.end_ms

        if delta >= 0:  # 오른쪽으로 확장
            if delta <= config.max_right_expand_ms:
                new_end = end_boundary
            else:
                new_end = chunk.end_ms + config.max_right_expand_ms
        else:  # 왼쪽으로 당김
            if abs(delta) <= config.max_right_shift_ms:
                new_end = end_boundary
            else:
                new_end = chunk.end_ms - config.max_right_shift_ms

        # 다음 청크와 겹치지 않도록
        if next_chunk is not None:
            new_end = min(new_end, next_chunk.start_ms)

    # 최소 길이 보장
    if new_end - new_start < config.min_entry_duration_ms:
        # 보정 취소
        new_start = chunk.start_ms
        new_end = chunk.end_ms

    return new_start, new_end


def apply_chunk_correction(
    chunk: Chunk,
    new_start: float,
    new_end: float,
    config: SyncConfig
) -> List[SubtitleEntry]:
    """청크 내부 엔트리들에 보정 적용

    Args:
        chunk: 원본 청크
        new_start: 새로운 시작 시간
        new_end: 새로운 끝 시간
        config: 싱크 설정

    Returns:
        보정된 엔트리 리스트
    """
    corrected_entries = []

    if config.entry_mode == "edge-only":
        # 엣지만 보정 모드: 내부 엔트리는 그대로, 경계만 클램핑
        for entry in chunk.entries:
            new_entry = SubtitleEntry(
                index=entry.index,
                start_ms=entry.start_ms,
                end_ms=entry.end_ms,
                text=entry.text
            )

            # 첫 엔트리: 시작 시간 클램핑
            if entry == chunk.entries[0]:
                new_entry.start_ms = max(new_entry.start_ms, new_start)
                new_entry.start_ms = min(new_entry.start_ms, new_start)
                new_entry.start_ms = new_start

            # 마지막 엔트리: 끝 시간 클램핑
            if entry == chunk.entries[-1]:
                new_entry.end_ms = min(new_entry.end_ms, new_end)
                new_entry.end_ms = max(new_entry.end_ms, new_end)
                new_entry.end_ms = new_end

            corrected_entries.append(new_entry)

    elif config.entry_mode == "proportional":
        # 비례 스케일링 모드
        old_duration = chunk.end_ms - chunk.start_ms
        new_duration = new_end - new_start

        if old_duration <= 0:
            # 안전 장치
            return [SubtitleEntry(
                index=e.index,
                start_ms=e.start_ms,
                end_ms=e.end_ms,
                text=e.text
            ) for e in chunk.entries]

        scale = new_duration / old_duration

        for entry in chunk.entries:
            # 청크 시작 기준 상대 시간
            rel_start = entry.start_ms - chunk.start_ms
            rel_end = entry.end_ms - chunk.start_ms

            # 스케일링 적용
            new_entry = SubtitleEntry(
                index=entry.index,
                start_ms=new_start + rel_start * scale,
                end_ms=new_start + rel_end * scale,
                text=entry.text
            )

            corrected_entries.append(new_entry)

    return corrected_entries


def fix_entry_overlaps(
    entries: List[SubtitleEntry],
    config: SyncConfig
) -> Tuple[List[SubtitleEntry], int]:
    """인접한 엔트리 간 겹침을 자동으로 보정

    Args:
        entries: 자막 엔트리 리스트 (index 순으로 정렬되어야 함)
        config: 싱크 설정

    Returns:
        (보정된 엔트리 리스트, 보정된 겹침 개수) 튜플
    """
    if not config.fix_overlaps or len(entries) < 2:
        return entries, 0

    # index 순으로 정렬
    sorted_entries = sorted(entries, key=lambda e: e.index)
    fixed_entries = []
    overlaps_fixed = 0

    for i, entry in enumerate(sorted_entries):
        # 현재 엔트리 복사
        current = SubtitleEntry(
            index=entry.index,
            start_ms=entry.start_ms,
            end_ms=entry.end_ms,
            text=entry.text
        )

        # 이전 엔트리가 있고, 현재 엔트리와 겹치는지 확인
        if i > 0:
            prev_entry = fixed_entries[-1]

            # 겹침 감지: 이전 엔트리의 끝이 현재 엔트리의 시작보다 늦음
            if prev_entry.end_ms > current.start_ms:
                overlaps_fixed += 1
                overlap_ms = prev_entry.end_ms - current.start_ms

                logger.debug(
                    f"겹침 감지 (엔트리 {prev_entry.index}-{current.index}): "
                    f"{overlap_ms:.0f}ms 겹침"
                )

                # 중간 지점 계산
                mid_point = (prev_entry.end_ms + current.start_ms) / 2

                # 최소 간격 적용
                half_gap = config.min_gap_ms / 2

                # 이전 엔트리의 끝을 중간점 - 간격/2로 설정
                new_prev_end = mid_point - half_gap
                # 현재 엔트리의 시작을 중간점 + 간격/2로 설정
                new_current_start = mid_point + half_gap

                # 최소 엔트리 길이 보장
                if new_prev_end - prev_entry.start_ms < config.min_entry_duration_ms:
                    # 이전 엔트리가 너무 짧아지는 경우
                    new_prev_end = prev_entry.start_ms + config.min_entry_duration_ms
                    new_current_start = new_prev_end + config.min_gap_ms

                if current.end_ms - new_current_start < config.min_entry_duration_ms:
                    # 현재 엔트리가 너무 짧아지는 경우
                    new_current_start = current.end_ms - config.min_entry_duration_ms
                    new_prev_end = new_current_start - config.min_gap_ms

                # 보정 적용
                prev_entry.end_ms = max(prev_entry.start_ms + config.min_entry_duration_ms, new_prev_end)
                current.start_ms = min(current.end_ms - config.min_entry_duration_ms, new_current_start)

                logger.debug(
                    f"겹침 보정 완료: 이전 엔트리 끝={prev_entry.end_ms:.0f}ms, "
                    f"현재 엔트리 시작={current.start_ms:.0f}ms, "
                    f"간격={current.start_ms - prev_entry.end_ms:.0f}ms"
                )

        fixed_entries.append(current)

    logger.info(f"겹침 보정: {overlaps_fixed}개 겹침 수정됨")
    return fixed_entries, overlaps_fixed


def sync_subtitles(
    subtitles: List[Dict],
    audio_path: str,
    config: Optional[SyncConfig] = None
) -> Tuple[List[Dict], Dict]:
    """자막을 VAD 기반으로 보정

    Args:
        subtitles: srt_module에서 파싱된 자막 리스트
        audio_path: 오디오 파일 경로
        config: 싱크 설정 (없으면 기본값)

    Returns:
        (보정된 자막 리스트, 통계 정보) 튜플
    """
    if config is None:
        config = SyncConfig()

    # 1. 자막 엔트리 파싱
    entries = parse_srt_entries(subtitles)
    if not entries:
        logger.warning("파싱된 엔트리가 없습니다.")
        return subtitles, {'status': 'no_entries'}

    # 2. VAD 세그먼트 추출
    logger.info("VAD 세그먼트 추출 중...")
    try:
        vad = SileroVAD(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            min_silence_duration_ms=config.vad_min_silence_duration_ms,
            speech_pad_ms=config.vad_speech_pad_ms
        )
        segments = vad.detect_speech_from_file(audio_path)
    except Exception as e:
        logger.error(f"VAD 처리 실패: {e}")
        return subtitles, {'status': 'vad_error', 'error': str(e)}

    # 세그먼트 후처리
    segments = merge_close_segments(segments, merge_threshold_ms=150)
    segments = filter_short_segments(segments, config.vad_min_speech_duration_ms)

    logger.info(f"VAD 세그먼트 {len(segments)}개 검출됨")

    if not segments:
        logger.warning("VAD 세그먼트가 없습니다.")
        return subtitles, {'status': 'no_segments'}

    # 3. 청크 구성
    chunks = build_chunks(entries, config.chunk_mode, config.gap_threshold_ms)

    # 4. 각 청크의 경계 보정
    corrected_entries = []
    corrections_applied = 0

    for i, chunk in enumerate(chunks):
        prev_chunk = chunks[i - 1] if i > 0 else None
        next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

        # 경계 스냅
        new_start, new_end = snap_chunk_boundaries(
            chunk,
            segments,
            config,
            prev_chunk,
            next_chunk
        )

        # 변경 여부 확인
        if abs(new_start - chunk.start_ms) > 1 or abs(new_end - chunk.end_ms) > 1:
            corrections_applied += 1
            logger.debug(
                f"청크 {i}: {chunk.start_ms:.0f}ms~{chunk.end_ms:.0f}ms -> "
                f"{new_start:.0f}ms~{new_end:.0f}ms"
            )

        # 내부 엔트리 처리
        chunk_corrected = apply_chunk_correction(chunk, new_start, new_end, config)
        corrected_entries.extend(chunk_corrected)

    # 5. 겹침 보정
    logger.info("엔트리 간 겹침 검사 및 보정 중...")
    corrected_entries, overlaps_fixed = fix_entry_overlaps(corrected_entries, config)

    # 6. 결과 생성
    result = []
    for entry in corrected_entries:
        result.append({
            'index': entry.index,
            'start': ms_to_time(entry.start_ms),
            'end': ms_to_time(entry.end_ms),
            'text': entry.text
        })

    stats = {
        'status': 'success',
        'total_entries': len(entries),
        'total_chunks': len(chunks),
        'vad_segments': len(segments),
        'corrections_applied': corrections_applied,
        'overlaps_fixed': overlaps_fixed
    }

    logger.info(
        f"보정 완료: {len(entries)}개 엔트리, {len(chunks)}개 청크, "
        f"{corrections_applied}개 청크 보정됨, {overlaps_fixed}개 겹침 수정됨"
    )

    return result, stats


def export_srt(subtitles: List[Dict]) -> str:
    """자막 리스트를 SRT 형식 문자열로 변환

    Args:
        subtitles: 자막 리스트

    Returns:
        SRT 형식 문자열
    """
    lines = []
    for sub in subtitles:
        lines.append(str(sub['index']))
        lines.append(f"{sub['start']} --> {sub['end']}")
        lines.append(sub['text'])
        lines.append('')  # 빈 줄

    return '\n'.join(lines)

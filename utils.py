import logging
import os
import json
import time
import re
from typing import Optional, Dict, List, Any
from dotenv import set_key
from datetime import datetime
import shutil

from constants import HISTORY_LOG_DIR, SNAPSHOT_ROOT_DIR, DEFAULT_CONTEXT_KEEP_RECENT, BASE_DIR
from module import ffmpeg_module
from module.gemini_module import GeminiClient

logger = logging.getLogger(__name__)

def _srt_timestamp_to_seconds(value: str) -> float:
    """SRT 타임스탬프(HH:MM:SS,mmm)를 초 단위 float로 변환합니다."""
    try:
        hours, minutes, rest = value.strip().split(":")
        seconds, millis = rest.split(",")
        total = (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000.0
        )
        return max(total, 0.0)
    except Exception:
        return 0.0

def _calculate_snapshot_count(entry_count: int) -> int:
    """청크 엔트리 수에 기반한 스냅샷 개수를 계산합니다.

    기본 1개, 10개 초과 시 10개당 1개 추가.
    예: 1-10개 → 1장, 11-20개 → 2장, 21-30개 → 3장, 50개 → 5장
    """
    return 1 + max(0, (entry_count - 10) // 10)

def _compute_chunk_time_bounds(subtitles: List[Dict[str, Any]]) -> tuple[float, float]:
    """청크 자막들의 시작/종료 시간을 초 단위로 계산합니다."""
    if not subtitles:
        return 0.0, 0.0
    starts = [_srt_timestamp_to_seconds(sub.get('start', '0')) for sub in subtitles]
    ends = [_srt_timestamp_to_seconds(sub.get('end', '0')) for sub in subtitles]
    start = max(0.0, min(starts))
    end = max(max(ends), start + 0.5)
    return start, end

def _make_chunk_timestamps(start: float, end: float, count: int, duration: float) -> List[float]:
    """청크 구간 내 균등 분포된 스냅샷 타임스탬프를 생성합니다."""
    if count <= 0:
        return []
    start = max(0.0, min(start, duration))
    end = min(max(end, start + 0.5), duration)
    if end - start < 0.5:
        padding = 0.25
        start = max(0.0, start - padding)
        end = min(duration, end + padding)
    span = max(end - start, 0.5)
    return [start + span * (i + 1) / (count + 1) for i in range(count)]

def _create_chunk_snapshots(stream_url: str, timestamps: List[float], chunk_index: int, session_dir: str) -> List[str]:
    """지정된 타임스탬프에서 스냅샷을 생성하고 경로 목록을 반환합니다."""
    if not timestamps:
        raise ValueError("스냅샷 타임스탬프가 비어 있습니다.")
    os.makedirs(session_dir, exist_ok=True)
    chunk_dir = os.path.join(session_dir, f"chunk_{chunk_index:04d}")
    os.makedirs(chunk_dir, exist_ok=True)
    prefix = f"chunk_{chunk_index:04d}"
    ffmpeg_module.snapshot_at_times(stream_url, timestamps, chunk_dir, prefix=prefix)
    return [os.path.join(chunk_dir, f"{prefix}_{idx:03d}.jpg") for idx in range(1, len(timestamps) + 1)]

def save_api_key_to_env(api_key: str) -> None:
    """사용자가 입력한 API 키를 .env(GOOGLE_API_KEY)와 환경변수에 저장합니다."""
    from constants import BASE_DIR
    ENV_PATH = os.path.join(BASE_DIR, '.env')
    api_key = (api_key or '').strip()
    if not api_key:
        return
    try:
        # .env에 저장 (GOOGLE_API_KEY만 사용)
        set_key(ENV_PATH, 'GOOGLE_API_KEY', api_key)
        # 현재 프로세스 환경에도 반영
        os.environ['GOOGLE_API_KEY'] = api_key
        logger.info(".env(GOOGLE_API_KEY)에 API 키가 저장되었습니다.")
    except Exception as e:
        logger.error(".env 저장 중 오류: %s", e)

def save_history_log(client: GeminiClient, job_id: Optional[str] = None) -> None:
    """최종 대화 히스토리를 로그 파일로 저장합니다."""
    try:
        os.makedirs(HISTORY_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_label = (job_id or 'session').strip()
        if job_label:
            job_label = re.sub(r'[^A-Za-z0-9_-]+', '-', job_label)[:80] or 'session'
        else:
            job_label = 'session'
        filename = f"history_{job_label}_{timestamp}.json"
        path = os.path.join(HISTORY_LOG_DIR, filename)
        history = client.get_history()
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(history, fp, ensure_ascii=False, indent=2)
        logger.info(
            "대화 히스토리가 로그 파일로 저장되었습니다: %s (메시지 %d개)",
            path,
            len(history),
        )
    except Exception as exc:
        logger.error("대화 히스토리 로그 저장 실패: %s", exc)


def cleanup_old_files(days: int = 3) -> None:
    """지정한 일수(days)보다 오래된 생성물(영상/음성/스냅샷/히스토리 로그)을 삭제합니다.

    - `generated_subtitles/` 하위 작업 디렉터리를 기준으로 삭제합니다.
    - `SNAPSHOT_ROOT_DIR`(스냅샷 세션 디렉터리)와 `HISTORY_LOG_DIR`(히스토리 로그 파일)도 정리합니다.
    이 함수는 안전성을 위해 개별 항목 삭제 실패를 무시하고 에러를 로깅합니다.
    """
    try:
        cutoff_ts = time.time() - float(days) * 86400.0

        # generated_subtitles 루트 (프로젝트 기준)
        from constants import BASE_DIR
        job_root = os.path.join(BASE_DIR, "generated_subtitles")
        if os.path.isdir(job_root):
            for entry in os.listdir(job_root):
                path = os.path.join(job_root, entry)
                try:
                    mtime = os.path.getmtime(path)
                    if mtime < cutoff_ts:
                        if os.path.isdir(path):
                            try:
                                shutil.rmtree(path)
                            except FileNotFoundError:
                                # 이미 삭제된 파일/폴더는 무시합니다.
                                logger.debug("이미 삭제된 작업 경로: %s", path)
                                continue
                        else:
                            try:
                                os.remove(path)
                            except FileNotFoundError:
                                logger.debug("이미 삭제된 작업 파일: %s", path)
                                continue
                        logger.info("오래된 작업 삭제: %s", path)
                except Exception as exc:
                    logger.exception("오래된 작업 삭제 실패: %s (%s)", path, exc)

        # snapshots 디렉터리 (세션별 폴더 삭제)
        try:
            snap_root = SNAPSHOT_ROOT_DIR
            if os.path.isdir(snap_root):
                for entry in os.listdir(snap_root):
                    path = os.path.join(snap_root, entry)
                    try:
                        mtime = os.path.getmtime(path)
                        if mtime < cutoff_ts:
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                            logger.info("오래된 스냅샷 삭제: %s", path)
                    except Exception as exc:
                        logger.exception("오래된 스냅샷 삭제 실패: %s (%s)", path, exc)
        except NameError:
            # SNAPSHOT_ROOT_DIR가 정의되어 있지 않으면 무시
            pass

        # 히스토리 로그 파일 정리
        try:
            logs_root = HISTORY_LOG_DIR
            if os.path.isdir(logs_root):
                for entry in os.listdir(logs_root):
                    path = os.path.join(logs_root, entry)
                    try:
                        mtime = os.path.getmtime(path)
                        if mtime < cutoff_ts:
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                            logger.info("오래된 로그 삭제: %s", path)
                    except Exception as exc:
                        logger.exception("오래된 로그 삭제 실패: %s (%s)", path, exc)
        except NameError:
            # HISTORY_LOG_DIR가 정의되어 있지 않으면 무시
            pass

    except Exception as exc:
        logger.exception("cleanup_old_files 실패: %s", exc)

def translate_srt_stream(content: str, client, target_lang: str = '한국어', batch_size: int = 10,
                         user_prompt: str = '', thinking_budget: int = 0, stop_flag = None,
                         video_context: Optional[Dict[str, Any]] = None,
                         job_label: Optional[str] = None):
    """SRT 텍스트를 번역하며 진행 상황을 스트림으로 제공합니다."""
    from module import srt_module
    from module.database_module import get_config_value

    target_lang = (target_lang or '한국어').strip()
    if not target_lang:
        target_lang = '한국어'
    logger.info("translate_srt_stream 호출됨 - 파라미터: target_lang=%s, batch_size=%s, thinking_budget=%s",
               target_lang, batch_size, thinking_budget)

    def build_translation_prompt(base_prompt: str, lang: str, entries: List[Dict[str, Any]]) -> str:
        """번역 요청 프롬프트를 생성합니다.

        Args:
            base_prompt (str): 사용자 지정 추가 프롬프트.
            lang (str): 목표 언어.
            entries (List[Dict]): {index, text} 필드를 가진 자막 엔트리 목록.

        Returns:
            str: 모델에 전달할 완성된 프롬프트 문자열.
        """
        safe_items: List[str] = []
        for sub in entries:
            # 기본 방어적 접근: key 누락 시 넘어감
            if 'index' not in sub or 'text' not in sub:
                continue
            # 기존 로직과 동일한 최소 escaping 유지 (쌍따옴표, 개행 정규화)
            escaped_text = str(sub['text']).replace('"', '\"').replace('\r', '').replace('\n', '\n')
            safe_items.append(json.dumps({"index": sub['index'], "text": escaped_text}, ensure_ascii=False))
        batch_json = '[' + ', '.join(safe_items) + ']'

        prompt_core = f"""{base_prompt}\n\n다음 자막들을 자연스러운 {lang}로 번역하세요.\n\n번역할 자막들:\n{batch_json}\n\n주의사항:\n- index는 원본과 동일하게 유지하세요\n- text만 번역하세요\n- 응답은 JSON 스키마에 정의된 형식을 정확히 따르세요\n"""
        return prompt_core

    # 응답 스키마 정의
    response_schema = {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "자막의 순서 번호"
                        },
                        "text": {
                            "type": "string",
                            "description": "번역된 자막 텍스트"
                        }
                    },
                    "required": ["index", "text"]
                }
            }
        },
        "required": ["translations"]
    }

    if not content.strip():
        yield json.dumps({'error': 'empty text'}) + "\n"
        return

    subtitles = srt_module.parse_srt_text(content)

    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = 10

    yield json.dumps({'count': len(subtitles)}) + "\n"

    processed_translations: Dict[int, str] = {}
    base_user_prompt = user_prompt or ''
    video_stream_url: Optional[str] = None
    video_duration: Optional[float] = None
    video_label: Optional[str] = None
    if video_context:
        video_stream_url = video_context.get('stream_url')
        duration_value = video_context.get('duration')
        try:
            video_duration = float(duration_value) if duration_value is not None else None
        except (TypeError, ValueError):
            video_duration = None
        video_label = video_context.get('youtube_url')
    chunk_counter = 0
    snapshot_session_dir: Optional[str] = None
    if video_stream_url and video_duration:
        base_label = job_label or video_label or 'session'
        safe_label = re.sub(r'[^A-Za-z0-9_-]+', '-', base_label) or 'session'
        snapshot_session_dir = os.path.join(
            SNAPSHOT_ROOT_DIR,
            f"{safe_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(snapshot_session_dir, exist_ok=True)

    for i in range(0, len(subtitles), batch_size):
        # 중단 플래그 확인
        if stop_flag and stop_flag.get('stopped', False):
            logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
            break

        batch = subtitles[i:i + batch_size]
        original_sub_map = {sub['index']: sub for sub in batch}
        chunk_counter += 1

        chunk_prompt_base = base_user_prompt
        chunk_media_paths: Optional[List[str]] = None
        chunk_context_text: Optional[str] = None

        if video_stream_url and video_duration and snapshot_session_dir:
            entry_count = len(batch)
            snapshot_count = _calculate_snapshot_count(entry_count)
            try:
                chunk_start, chunk_end = _compute_chunk_time_bounds(batch)
                timestamps = _make_chunk_timestamps(chunk_start, chunk_end, snapshot_count, video_duration)
                snapshot_paths = _create_chunk_snapshots(
                    video_stream_url,
                    timestamps,
                    chunk_counter,
                    snapshot_session_dir,
                )
                analysis_prompt = (
                    f"다음 이미지는 유튜브 영상 청크 #{chunk_counter}"
                    f" (엔트리 {entry_count}개)"
                    f"{f' - {video_label}' if video_label else ''}에 해당하는 장면입니다. "
                    "영상의 주요 등장인물, 사건, 분위기를 한국어로 요약하고, "
                    "번역 시 주의해야 할 맥락이나 용어를 정리해 주세요."
                )
                chunk_media_paths = snapshot_paths
                chunk_context_text = (
                    f"[영상 청크 #{chunk_counter} 이미지 컨텍스트]\n"
                    f"- 엔트리 수: {entry_count}\n"
                    f"- 원본 영상: {video_label or video_stream_url}\n"
                    "위 스냅샷을 참고하여 주요 등장인물, 장면 분위기, 시각적 단서를 고려한 번역 결과를 생성하세요."
                )
                logger.info(
                    "청크 %s 이미지 %s장 경로 준비 완료",
                    chunk_counter,
                    len(chunk_media_paths),
                )
            except Exception as exc:
                logger.error("청크 %s 스냅샷 처리 실패: %s", chunk_counter, exc)
                chunk_media_paths = None
                chunk_context_text = None

        prompt = build_translation_prompt(chunk_prompt_base, target_lang, batch)
        if chunk_context_text:
            prompt = f"{chunk_context_text}\n\n{prompt}"

        # 새 통합 방식: 매 배치 전에 GeminiClient에 재구성 자막 블록만 업데이트.
        # 실제 [WORK SUMMARY] 생성/압축 시점은 GeminiClient 내부 로직이 판단.
        already_processed = subtitles[:i]
        keep_recent_entries = max(0, int(getattr(client, 'context_keep_recent', DEFAULT_CONTEXT_KEEP_RECENT) or 0))
        # 조건: 히스토리(현재 + 이번 배치 프롬프트 예상)가 context_limit_tokens를 초과할 때만 스냅샷 재구성 실행
        try:
            limit_tokens = getattr(client, 'context_limit_tokens', None)
            compression_enabled = getattr(client, 'context_compression_enabled', False)
            should_snapshot = False
            if compression_enabled and limit_tokens:
                try:
                    current_hist_tokens = client._estimate_history_tokens()  # 내부 휴리스틱 사용
                    upcoming_tokens = client._estimate_text_tokens(prompt)
                    if current_hist_tokens + upcoming_tokens > limit_tokens:
                        should_snapshot = True
                except Exception:
                    # 추정 실패 시 보수적으로 스냅샷 생략
                    pass

            if should_snapshot:
                logger.debug(
                    "컨텍스트 압축 활성 & 한도 초과 예상 -> 히스토리 스냅샷 재구성 실행 (current+upcoming > limit)"
                )
                client.set_reconstructed_subtitles(
                    already_processed,
                    processed_translations,
                    keep_recent_entries=keep_recent_entries,
                    as_individual_messages=True,
                )
        except Exception as exc:
            logger.debug("조건부 set_reconstructed_subtitles 처리 중 오류: %s", exc)

        retry_count = 0
        max_retries = 5
        batch_success = False
        confirmed_translations: Dict[int, str] = {}
        current_batch_to_translate = list(batch)  # 현재 시도할 배치

        while retry_count < max_retries and not batch_success:
            # 중단 플래그 확인
            if stop_flag and stop_flag.get('stopped', False):
                logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
                return

            # 재시도 시 미처리된 엔트리만 포함한 새 배치 생성
            if retry_count > 0:
                unprocessed_indices = set(original_sub_map.keys()) - set(confirmed_translations.keys())
                if not unprocessed_indices:
                    # 모든 엔트리가 처리됨
                    batch_success = True
                    break

                current_batch_to_translate = [original_sub_map[idx] for idx in unprocessed_indices]
                logger.info(f"재시도 {retry_count}회: 미처리 엔트리 {len(current_batch_to_translate)}개만 재번역")

                # 미처리 엔트리만 포함한 새 프롬프트 생성
                prompt = build_translation_prompt(chunk_prompt_base, target_lang, current_batch_to_translate)

            try:
                buffer = ""
                processed_indices = set()
                batch_translations: Dict[int, str] = {}

                # response_schema를 사용하여 스트리밍 요청 (통합 메서드 사용)
                stream_iterator = client.send_message_stream(
                    prompt,
                    response_schema=response_schema,
                    media_paths=chunk_media_paths,
                )

                for chunk in stream_iterator:
                    # 각 청크마다 중단 플래그 확인
                    if stop_flag and stop_flag.get('stopped', False):
                        logger.info("스트리밍 중 중단 플래그가 설정되어 번역을 중지합니다.")
                        return

                    buffer += chunk

                    # translations 배열 내의 개별 객체들을 실시간으로 파싱
                    while True:
                        # "index"를 포함한 객체 찾기
                        index_pos = buffer.find('"index"')
                        if index_pos == -1:
                            break

                        # 해당 객체의 시작점 찾기
                        object_start = buffer.rfind('{', 0, index_pos)
                        if object_start == -1:
                            break

                        brace_level = 0
                        object_end = -1
                        for i in range(object_start, len(buffer)):
                            if buffer[i] == '{':
                                brace_level += 1
                            elif buffer[i] == '}':
                                brace_level -= 1

                            if brace_level == 0:
                                object_end = i
                                break

                        if object_end == -1:
                            break

                        obj_str = buffer[object_start : object_end + 1]

                        try:
                            item = json.loads(obj_str)
                            item_index = item.get("index")
                            original_sub = original_sub_map.get(item_index)

                            if original_sub and item_index not in processed_indices and item_index not in confirmed_translations:
                                result = {
                                    'index': item_index,
                                    'start': original_sub['start'],
                                    'end': original_sub['end'],
                                    'original': original_sub['text'],
                                    'translated': item.get('text', original_sub['text'])
                                }
                                yield json.dumps(result, ensure_ascii=False) + "\n"
                                processed_indices.add(item_index)
                                batch_translations[item_index] = result['translated']

                            buffer = buffer[object_end + 1:]

                        except json.JSONDecodeError:
                            break

                # 이번 시도에서 파싱된 엔트리를 confirmed에 추가
                confirmed_translations.update(batch_translations)

                # 원본 배치의 모든 엔트리가 처리되었는지 확인
                if len(confirmed_translations) == len(batch):
                    batch_success = True
                else:
                    raise ValueError(f"스트림 처리 중 일부 자막이 누락되었습니다. (처리됨: {len(confirmed_translations)}/{len(batch)})")

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.error(f"번역 재시도 {retry_count}회: {e} (이미 처리됨: {len(confirmed_translations)}/{len(batch)})")
                    try:
                        client.delete_last_turn()
                    except Exception:
                        pass
                    time.sleep(10)
                else:
                    error_message = f"배치 처리 실패 (재시도 {max_retries}회 초과): {e}"
                    unprocessed_indices = set(original_sub_map.keys()) - set(confirmed_translations.keys())
                    for index in unprocessed_indices:
                        sub = original_sub_map[index]
                        error_result = {
                            'index': index,
                            'start': sub['start'],
                            'end': sub['end'],
                            'original': sub['text'],
                            'translated': f'-- 번역 실패: {e} --',
                            'error': True
                        }
                        yield json.dumps(error_result, ensure_ascii=False) + "\n"
                        confirmed_translations[index] = error_result['translated']
                    logger.error(error_message)
                    batch_success = True

        if confirmed_translations:
            processed_translations.update(confirmed_translations)

            # 재시도로 인해 히스토리의 마지막 턴이 부분 요청/응답만 포함할 수 있음
            # 전체 배치를 반영한 완전한 턴으로 히스토리 재구성
            if retry_count > 0 and len(client.history) >= 2:
                if client.history[-2].get('role') == 'user' and client.history[-1].get('role') == 'model':
                    # 원본 전체 배치의 프롬프트 재구성
                    original_batch_items = []
                    for idx in sorted(original_sub_map.keys()):
                        sub = original_sub_map[idx]
                        escaped_text = sub["text"].replace('"', '\"').replace('\n', '\n').replace('\r', '')
                        original_batch_items.append(json.dumps({"index": sub["index"], "text": escaped_text}))

                    original_batch_json = '[' + ', '.join(original_batch_items) + ']'
                    # 재구성 시에도 동일한 helper 사용 (원본 배치 기준)
                    # original_batch_items 구성은 위에서 유지
                    original_entries_for_prompt = [
                        { 'index': original_sub_map[idx]['index'], 'text': original_sub_map[idx]['text'] }
                        for idx in sorted(original_sub_map.keys())
                    ]
                    complete_prompt = build_translation_prompt(chunk_prompt_base, target_lang, original_entries_for_prompt)

                    # 전체 배치의 완전한 JSON 응답 재구성
                    complete_translations = []
                    for idx in sorted(confirmed_translations.keys()):
                        complete_translations.append({
                            "index": idx,
                            "text": confirmed_translations[idx]
                        })

                    complete_response = json.dumps(
                        {"translations": complete_translations},
                        ensure_ascii=False
                    )

                    # 히스토리의 마지막 턴을 완전한 요청/응답으로 교체
                    request_parts = client.history[-2].setdefault('parts', [])
                    if not isinstance(request_parts, list):
                        request_parts = client.history[-2]['parts'] = []
                    replaced_request = False
                    for part in request_parts:
                        if isinstance(part, dict) and 'text' in part:
                            part['text'] = complete_prompt
                            replaced_request = True
                            break
                    if not replaced_request:
                        request_parts.insert(0, {'text': complete_prompt})

                    response_parts = client.history[-1].setdefault('parts', [])
                    if not isinstance(response_parts, list):
                        response_parts = client.history[-1]['parts'] = []
                    replaced_response = False
                    for part in response_parts:
                        if isinstance(part, dict) and 'text' in part:
                            part['text'] = complete_response
                            replaced_response = True
                            break
                    if not replaced_response:
                        response_parts.insert(0, {'text': complete_response})
                    logger.info(f"재시도 후 히스토리 재구성 완료: 전체 {len(confirmed_translations)}개 엔트리 반영")

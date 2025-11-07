import logging
import os
import tempfile
import json
import time
import uuid
from datetime import datetime
import re
from typing import Optional, Dict, List, Any
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv, set_key
from module import srt_module, ffmpeg_module
from module.gemini_module import GeminiClient
from module.database_module import (
    list_presets,
    get_preset,
    save_preset,
    delete_preset,
    save_job,
    get_job,
    delete_job,
    delete_old_jobs,
    get_all_config,
    set_configs,
    get_config_value,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# .env 로드 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CONTEXT_KEEP_RECENT = 50  # 히스토리 재구성 시 유지할 최근 자막 엔트리 수
HISTORY_LOG_DIR = os.path.join(BASE_DIR, 'logs')
SNAPSHOT_ROOT_DIR = os.path.join(BASE_DIR, 'snapshots')

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


## 레거시 재구성 헬퍼 제거됨: GeminiClient.set_reconstructed_subtitles 로 일원화

def translate_srt_stream(content: str, client, target_lang: str = '한국어', batch_size: int = 10,
                         user_prompt: str = '', thinking_budget: int = 0, stop_flag = None,
                         video_context: Optional[Dict[str, Any]] = None,
                         job_label: Optional[str] = None):
    """SRT 텍스트를 번역하며 진행 상황을 스트림으로 제공합니다."""
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

@app.route('/')
def index():
    missing_api_key = not bool(os.environ.get('GOOGLE_API_KEY', '').strip())
    return render_template('index.html', missing_api_key=missing_api_key)

@app.route('/progress')
def progress():
    return render_template('progress.html')


@app.route('/subtitle_generate')
def subtitle_generate():
    """자막 생성 페이지 템플릿을 렌더링합니다."""
    missing_api_key = not bool(os.environ.get('GOOGLE_API_KEY', '').strip())
    return render_template('subtitle_generate.html', missing_api_key=missing_api_key)


@app.route('/api/settings/api-key', methods=['POST'])
def api_save_google_api_key():
    data = request.get_json(silent=True) or {}
    api_key = (data.get('api_key') or '').strip()
    if not api_key:
        return jsonify({'error': 'API 키를 입력해 주세요.'}), 400
    save_api_key_to_env(api_key)
    # config 테이블에도 반영
    set_configs({'api_key': api_key})
    return jsonify({'status': 'saved'})

# ----- Config Management (추가 설정; 프리셋과 명확히 분리) -----
@app.route('/api/config', methods=['GET'])
def api_get_config():
    """추가 설정 전체 조회. api_key는 실제 값을 노출하지 않고 preview만 제공."""
    data = get_all_config()
    out = data.copy()
    if 'api_key' in out and isinstance(out['api_key'], str):
        val = out['api_key']
        if len(val) > 8:
            out['api_key_preview'] = val[:4] + '...' + val[-4:]
        else:
            out['api_key_preview'] = '***'
        out.pop('api_key')
    return jsonify(out)

@app.route('/api/config', methods=['POST'])
def api_update_config():
    data = request.get_json(silent=True) or {}
    updates = {}
    # API Key
    if 'api_key' in data:
        new_key = (data.get('api_key') or '').strip()
        if new_key:
            save_api_key_to_env(new_key)
            updates['api_key'] = new_key
    # 모델명
    if 'model_name' in data:
        model_name = (data.get('model_name') or '').strip()
        if model_name:
            updates['model_name'] = model_name
    # RPM 제한
    if 'rpm_limit' in data:
        raw_rl = data.get('rpm_limit')
        try:
            rl = int(str(raw_rl))
            if rl > 0:
                updates['rpm_limit'] = rl
        except (ValueError, TypeError):
            pass
    # 전역 기본 thinking_budget (개별 요청은 override 가능)
    if 'thinking_budget' in data:
        raw_tb = data.get('thinking_budget')
        try:
            tb = int(str(raw_tb))
            if tb > 0:
                updates['thinking_budget'] = tb
        except (ValueError, TypeError):
            pass
    # 컨텍스트 압축 on/off
    if 'context_compression' in data:
        raw = str(data.get('context_compression')).lower()
        updates['context_compression'] = 1 if raw in {'1','true','on'} else 0
    # 컨텍스트 토큰 제한
    if 'context_limit' in data:
        raw_cl = data.get('context_limit')
        try:
            cl = int(str(raw_cl))
            if cl > 0:
                updates['context_limit'] = cl
        except (ValueError, TypeError):
            pass
    if not updates:
        return jsonify({'status': 'no-op'}), 400
    set_configs(updates)
    safe = updates.copy()
    if 'api_key' in safe:
        safe['api_key'] = 'stored'
    return jsonify({'status': 'saved', 'updated': safe})

@app.route('/api/jobs', methods=['POST'])
def api_create_job():
    """업로드된 SRT 데이터와 옵션을 임시로 저장하고 작업 ID를 반환합니다."""
    # 새로운 작업을 추가하기 전에 오래된 작업(30일 경과)을 정리합니다.
    delete_old_jobs()
    
    # SRT 파일 데이터 수집: 업로드된 파일 또는 폼 텍스트를 리스트로 저장
    files_data = []
    # 업로드된 파일들 처리 (다중 파일 지원)
    for f in request.files.getlist('srt_files'):
        if f and f.filename:
            files_data.append({
                'name': f.filename,
                'text': f.read().decode('utf-8')
            })

    # 파일이 없으면 폼 데이터에서 SRT 텍스트를 가져옴
    if not files_data:
        text = request.form.get('srt_text', '')
        if text.strip():
            files_data.append({'name': 'pasted.srt', 'text': text})

    # SRT 데이터가 없으면 에러 반환
    if not files_data:
        return jsonify({'error': 'no srt data'}), 400

    # 고유 작업 ID 생성
    job_id = str(uuid.uuid4())
    # 배치 크기 파라미터 처리 (기본값 10)
    batch_size = request.form.get('batch_size', 10)
    try:
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        batch_size = 10
    # Thinking Budget 파라미터 처리 (기본값 0)
    thinking_budget = request.form.get('thinking_budget', 0)
    try:
        thinking_budget = int(thinking_budget)
    except (ValueError, TypeError):
        thinking_budget = 0
    # 컨텍스트 압축 설정 처리
    raw_context_compression = (request.form.get('context_compression') or '').strip()
    context_compression = 1 if raw_context_compression in {'1', 'true', 'True', 'on'} else 0
    # 컨텍스트 토큰 제한 처리
    context_limit = request.form.get('context_limit')
    try:
        context_limit = int(context_limit) if context_limit not in (None, '') else None
    except (ValueError, TypeError):
        context_limit = None
    # API 키 처리 (제공 시 .env에 저장)
    api_key = (request.form.get('api_key') or '').strip()
    if api_key:
        # 사용자가 제출한 키를 즉시 .env에 저장하고 환경에 반영
        save_api_key_to_env(api_key)
    # 모델 이름 처리 (기본값 DEFAULT_MODEL)
    model_name = (request.form.get('model') or '').strip() or DEFAULT_MODEL
    youtube_url = (request.form.get('youtube_url') or '').strip()
    # 작업 데이터 구성
    job_data = {
        'files': files_data,
        'target_lang': request.form.get('target_lang', '한국어'),
        'batch_size': batch_size,
        'custom_prompt': request.form.get('custom_prompt', ''),
        'thinking_budget': thinking_budget,
        'context_compression': context_compression,
        'context_limit': context_limit,
        'model': model_name,
        'youtube_url': youtube_url,
    }
    # 작업 데이터를 DB에 저장
    save_job(job_id, job_data) # type: ignore
    # 생성된 작업 ID 반환
    return jsonify({'job_id': job_id})

@app.route('/api/jobs/<job_id>', methods=['GET'])
def api_get_job(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify(job)

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def api_delete_job(job_id):
    delete_job(job_id)
    return jsonify({'status': 'deleted'})

@app.route('/upload_srt', methods=['POST'])
def upload_srt():
    """SRT 파일 업로드를 처리하고, 자막을 파싱하여 실시간으로 번역합니다."""
    logger.info("Upload SRT 요청 폼 데이터: %s", dict(request.form))
    
    files = request.files.getlist('srt_files')
    if not files:
        file = request.files.get('srt_file')
        if file:
            files = [file]
    if not files:
        return jsonify({'error': 'file missing'}), 400

    temp_paths = []
    for f in files:
        if f.filename == '':
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.srt')
        try:
            f.save(tmp.name)
        finally:
            tmp.close()
        temp_paths.append((f.filename, tmp.name))

    # 파일명 기준으로 오름차순 정렬
    temp_paths.sort(key=lambda x: x[0])

    logger.info("처리할 SRT 파일들: %s", [name for name, _ in temp_paths])

    target_lang = request.form.get('target_lang', '한국어')
    batch_size = request.form.get('batch_size', 10)
    try:
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        batch_size = 10
    user_prompt = request.form.get('custom_prompt', '')
    youtube_url = (request.form.get('youtube_url') or '').strip()
    thinking_budget = request.form.get('thinking_budget', 0)
    api_key = (request.form.get('api_key') or '').strip()
    model_name = (request.form.get('model') or '').strip() or DEFAULT_MODEL
    job_id = (request.form.get('job_id') or '').strip() or None
    raw_context_compression = (request.form.get('context_compression') or '').strip()
    context_compression_enabled = raw_context_compression in {'1', 'true', 'True', 'on'}
    context_limit = request.form.get('context_limit')
    try:
        context_limit = int(context_limit) if context_limit not in (None, '') else None
    except (ValueError, TypeError):
        context_limit = None
    # 사용자가 업로드 단계에서 키를 다시 보냈다면 .env에 저장/반영
    if api_key:
        save_api_key_to_env(api_key)

    # 중단 플래그 생성
    stop_flag = {'stopped': False}

    # 모든 파일에서 공유할 단일 클라이언트 생성
    try:
        thinking_budget_val = int(thinking_budget)
    except (ValueError, TypeError):
        logger.warning("thinking_budget 파라미터 변환 실패, 기본값 0 사용: %s", thinking_budget)
        thinking_budget_val = 0

    # -1은 auto를 의미하며, thinking_config를 설정하지 않음 (Gemini가 자동 결정)
    generation_config_extra = {
        'max_output_tokens': 122880,  # 120k 토큰
    }

    if thinking_budget_val == -1:
        logger.info("Thinking Budget이 auto로 설정되었습니다. Gemini가 자동 추론 모드를 사용합니다.")
        generation_config_extra['thinking_config'] = {'thinking_budget': thinking_budget_val}
    elif thinking_budget_val > 0:
        generation_config_extra['thinking_config'] = {'thinking_budget': thinking_budget_val}

    try:
        shared_client = GeminiClient(
            model=model_name,
            api_key=api_key if api_key else None,
            thinking_budget=thinking_budget_val,
            rpm_limit=int(get_config_value('rpm_limit', 9) or 9),
            generation_config=generation_config_extra,
            context_compression_enabled=context_compression_enabled,
            context_limit_tokens=context_limit,
            context_keep_recent=DEFAULT_CONTEXT_KEEP_RECENT,
        )
    except ValueError as exc:
        logger.error("Gemini 클라이언트 초기화 실패: %s", exc)
        for _, p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        return jsonify({'error': str(exc)}), 400
    if thinking_budget_val == -1:
        logger.info("Thinking Budget: auto (Gemini 자동 추론, -1 전달)")
    elif thinking_budget_val <= 0:
        logger.info("Thinking 기능 비활성화됨 (thinking_budget=0)")
    else:
        logger.info(f"Thinking Budget 설정: {thinking_budget_val}")
    if context_compression_enabled:
        logger.info(
            "컨텍스트 압축 활성화. limit=%s, keep_recent=%s",
            context_limit if context_limit is not None else '미설정',
            DEFAULT_CONTEXT_KEEP_RECENT,
        )
    
    shared_client.start_chat()
    
    logger.info("다중 파일 처리를 위한 공유 클라이언트가 생성되었습니다.")

    video_context: Optional[Dict[str, Any]] = None
    if youtube_url:
        try:
            stream_url = ffmpeg_module.get_stream_url(youtube_url)
            duration_seconds = ffmpeg_module.get_duration_seconds(stream_url)
            video_context = {
                'youtube_url': youtube_url,
                'stream_url': stream_url,
                'duration': duration_seconds,
            }
            logger.info(
                "유튜브 영상 컨텍스트 준비 완료: duration=%.2fs",
                duration_seconds,
            )
        except Exception as exc:
            logger.error("유튜브 스트림 정보 준비 실패: %s", exc)
            video_context = None

    def generate():
        try:
            # 총 파일 수 전송
            yield json.dumps({'total_files': len(temp_paths)}) + "\n"
            
            for name, path in temp_paths:
                # 중단 플래그 확인
                if stop_flag['stopped']:
                    logger.info("중단 플래그가 설정되어 파일 처리를 중지합니다.")
                    break
                    
                # 현재 처리 중인 파일 정보 전송
                yield json.dumps({'current_file': name}) + "\n"
                
                with open(path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                # 파일별 번역 결과를 스트림으로 전송 (공유 클라이언트 사용)
                for line in translate_srt_stream(
                    srt_content,
                    shared_client,
                    target_lang,
                    batch_size,
                    user_prompt,
                    thinking_budget_val,
                    stop_flag,
                    video_context,
                    job_id,
                ):
                    yield line
                    # 중단 플래그 확인
                    if stop_flag['stopped']:
                        logger.info("중단 플래그가 설정되어 스트림 전송을 중지합니다.")
                        break
                
                # 파일 완료 신호
                if not stop_flag['stopped']:
                    yield json.dumps({'file_completed': name}) + "\n"

            if not stop_flag['stopped']:
                save_history_log(shared_client, job_id)
        except GeneratorExit:
            logger.warning("클라이언트 연결이 중단되었습니다. 번역 작업을 중지합니다.")
            stop_flag['stopped'] = True
        finally:
            # 임시 파일들 정리
            for _, p in temp_paths:
                try:
                    os.remove(p)
                except:
                    pass

    return Response(generate(), mimetype='text/plain')

@app.route('/parse_srt', methods=['POST'])
def parse_srt_route():
    """클라이언트에서 전송된 SRT 텍스트를 파싱하여 전체 자막 배열을 반환합니다."""
    data = request.get_json() or {}
    content = data.get('srt_text', '')
    if not content.strip():
        return jsonify({'error': 'empty text'}), 400

    subtitles = srt_module.parse_srt_text(content)
    return jsonify(subtitles)

# ----- Preset management API -----

@app.route('/api/presets', methods=['GET'])
def api_list_presets():
    """Return all stored presets."""
    return jsonify(list_presets())

@app.route('/api/presets/<name>', methods=['GET'])
def api_get_preset(name):
    preset = get_preset(name)
    if preset:
        # (요청에 따라 thinking_budget은 프리셋에 포함해 저장/로드 가능하도록 허용)
        thinking_budget_value = preset.get('thinking_budget')
        # DB에서 -1로 저장된 경우 'auto'로 변환
        if thinking_budget_value == -1:
            thinking_budget_value = 'auto'

        result = {
            'name': preset.get('name'),
            'target_lang': preset.get('target_lang'),
            'batch_size': preset.get('batch_size'),
            'custom_prompt': preset.get('custom_prompt'),
            'thinking_budget': thinking_budget_value,  # 0, 'auto', 또는 None 가능
        }
        return jsonify(result)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/presets/<name>', methods=['POST'])
def api_save_preset(name):
    data = request.get_json() or {}
    # 프리셋은 번역 기본 파라미터를 저장합니다.
    # thinking_budget은 요청에 따라 프리셋 단위 저장을 허용 (글로벌 기본값과 별개로 오버라이드 용도)
    target_lang = data.get('target_lang')
    batch_size_raw = data.get('batch_size')
    try:
        batch_size = int(batch_size_raw) if batch_size_raw is not None else None
    except (ValueError, TypeError):
        batch_size = None
    custom_prompt = data.get('custom_prompt')
    # thinking_budget 처리 (0, -1(auto), 또는 양의 정수)
    thinking_budget_val = None
    if 'thinking_budget' in data:
        try:
            tb_raw = data.get('thinking_budget')
            if tb_raw is not None and tb_raw != '':
                # -1은 auto를 의미하며 DB에 그대로 저장
                tb_int = int(tb_raw)
                if tb_int >= -1:  # -1 이상의 값만 허용 (-1, 0, 양수)
                    thinking_budget_val = tb_int
        except (ValueError, TypeError):
            thinking_budget_val = None

    # legacy 파라미터( api_key, context_* )는 저장하지 않음.
    save_preset(
        name,
        target_lang,
        batch_size,
        custom_prompt,
        thinking_budget=thinking_budget_val,
        api_key=None,
        context_compression=None,
        context_limit=None,
    )
    return jsonify({'status': 'ok', 'stored_fields': ['target_lang', 'batch_size', 'custom_prompt', 'thinking_budget']})

@app.route('/api/presets/<name>', methods=['DELETE'])
def api_delete_preset(name):
    delete_preset(name)
    return jsonify({'status': 'deleted'})

if __name__ == '__main__':
    app.run(debug=True)

import logging
import os
import tempfile
import json
import time
import uuid
from datetime import datetime
import re
from typing import Optional, Dict, List, Any, Tuple
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv, set_key
from module import srt_module
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
DEFAULT_CONTEXT_KEEP_RECENT = 100  # 히스토리 재구성 시 유지할 최근 자막 엔트리 수
HISTORY_LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 업로드된 파일과 옵션을 임시로 보관하는 작업 저장소 (DB로 대체됨)
# pending_jobs: dict[str, dict] = {}


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


def _truncate_for_context(text: str, max_length: int = 400) -> str:
    """재구성 히스토리에 포함될 텍스트를 간결하게 만듭니다."""
    if not text:
        return ''
    collapsed = re.sub(r'\s+', ' ', text).strip()
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[:max_length].rstrip() + '...'


def _clone_history_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """start_chat 호출용 히스토리 메시지를 복사합니다."""
    cloned: List[Dict[str, Any]] = []
    for message in messages:
        parts = [{'text': part.get('text', '')} for part in (message.get('parts') or [])]
        cloned.append({'role': message.get('role', 'model'), 'parts': parts})
    return cloned


def _find_context_summary_message(client: GeminiClient) -> Optional[Dict[str, Any]]:
    """히스토리에서 최신 [컨텍스트 요약] 메시지를 찾아 복사합니다."""
    prefix = getattr(client, '_context_summary_prefix', None)
    if not prefix:
        return None
    try:
        history = client.get_history()
    except Exception:
        history = []

    for message in reversed(history):
        if message.get('role') != 'model':
            continue
        parts = message.get('parts') or []
        for part in parts:
            text = (part.get('text') or '').strip()
            if text.startswith(prefix):
                return _clone_history_messages([message])[0]
    return None


def _build_reconstructed_summary(entries: List[Dict[str, Any]], total_count: int) -> str:
    """토큰 제한에 걸릴 때 사용할 요약 텍스트를 생성합니다."""
    lines = ["[RECONSTRUCTED SUBTITLES]"]
    if not entries:
        lines.append("이전에 처리된 자막이 없거나 요약할 수 없습니다.")
        return "\n".join(lines)

    lines.append(f"최근 {len(entries)}개의 자막 번역 요약")
    if total_count > len(entries):
        lines.append(f"(총 {total_count}개 중 최신 {len(entries)}개)")

    for item in entries:
        index = item.get('index')
        translated = _truncate_for_context(item.get('translated', ''), 160)
        lines.append(f"#{index}: {translated}")

    return "\n".join(lines)


def _prepare_reconstructed_history(
    client: GeminiClient,
    processed_subtitles: List[Dict[str, Any]],
    translations: Dict[int, str],
    keep_recent_entries: int,
    upcoming_prompt: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """최근 처리된 자막을 기반으로 Gemini 히스토리를 재구성합니다."""
    metadata = {
        'total_entries': 0,
        'recent_entries': 0,
        'history_tokens': 0,
        'mode': 'empty',
        'keep_requested': keep_recent_entries,
    }

    if keep_recent_entries <= 0 or not processed_subtitles or not translations:
        return [], metadata

    reconstructed: List[Dict[str, Any]] = []
    for subtitle in processed_subtitles:
        translated = translations.get(subtitle.get('index')) # pyright: ignore[reportArgumentType]
        if translated is None:
            continue
        reconstructed.append({
            'index': subtitle.get('index'),
            'start': subtitle.get('start'),
            'end': subtitle.get('end'),
            'original': _truncate_for_context(subtitle.get('text', '')),
            'translated': _truncate_for_context(translated),
        })

    if not reconstructed:
        return [], metadata

    total_count = len(reconstructed)
    recent_entries = reconstructed[-keep_recent_entries:]
    trimmed_count = total_count - len(recent_entries)

    payload = json.dumps(recent_entries, ensure_ascii=False, separators=(',', ':'))
    header_lines = ["[RECONSTRUCTED SUBTITLES]"]
    if trimmed_count > 0:
        header_lines.append(
            f"총 {total_count}개 중 최신 {len(recent_entries)}개 자막만 유지합니다."
        )
    else:
        header_lines.append(f"최근 {len(recent_entries)}개의 자막 번역 기록입니다.")

    history_text = "\n".join(header_lines) + "\n" + payload

    estimate_tokens = getattr(client, '_estimate_text_tokens')
    context_limit = getattr(client, 'context_limit_tokens', None)
    mode = 'full'
    if context_limit:
        history_tokens = estimate_tokens(history_text)
        prompt_tokens = estimate_tokens(upcoming_prompt)
        safety_buffer = 256
        if history_tokens + prompt_tokens + safety_buffer > context_limit:
            history_text = _build_reconstructed_summary(recent_entries, total_count)
            history_tokens = estimate_tokens(history_text)
            mode = 'summary'
            if history_tokens + prompt_tokens + safety_buffer > context_limit:
                history_text = (
                    "[RECONSTRUCTED SUBTITLES]\n"
                    "최근 자막 기록이 길어 요약이 필요합니다. 이전 배치들의 번역 맥락은 유지되었다고 가정하고 일관성을 유지해 주세요."
                )
                history_tokens = estimate_tokens(history_text)
                mode = 'fallback'
    else:
        history_tokens = estimate_tokens(history_text)

    logger.debug(
        "재구성 히스토리 준비: total=%s, recent=%s, keep_recent=%s",
        len(reconstructed),
        len(recent_entries),
        keep_recent_entries,
    )

    metadata.update({
        'total_entries': total_count,
        'recent_entries': len(recent_entries),
        'history_tokens': history_tokens,
        'mode': mode,
    })

    return [{
        'role': 'model',
        'parts': [{'text': history_text}]
    }], metadata

def translate_srt_stream(content: str, client, target_lang: str = '한국어', batch_size: int = 10, user_prompt: str = '', thinking_budget: int = 8192, stop_flag = None):
    """SRT 텍스트를 번역하며 진행 상황을 스트림으로 제공합니다."""
    logger.info("translate_srt_stream 호출됨 - 파라미터: target_lang=%s, batch_size=%s, thinking_budget=%s",
               target_lang, batch_size, thinking_budget)
    
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

    for i in range(0, len(subtitles), batch_size):
        # 중단 플래그 확인
        if stop_flag and stop_flag.get('stopped', False):
            logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
            break
            
        batch = subtitles[i:i + batch_size]
        original_sub_map = {sub['index']: sub for sub in batch}

        batch_items = []
        for sub in batch:
            escaped_text = sub["text"].replace('"', '\"').replace('\n', '\n').replace('\r', '')
            batch_items.append(json.dumps({"index": sub["index"], "text": escaped_text}))

        batch_json = '[' + ', '.join(batch_items) + ']'

        prompt = f"""{user_prompt}
다음 자막들을 자연스러운 {target_lang}로 번역하세요.

번역할 자막들:
{batch_json}

주의사항:
- index는 원본과 동일하게 유지하세요
- text만 번역하세요
- 응답은 JSON 스키마에 정의된 형식을 정확히 따르세요
"""

        already_processed = subtitles[:i]
        keep_recent_entries = max(
            0,
            int(getattr(client, 'context_keep_recent', DEFAULT_CONTEXT_KEEP_RECENT) or 0)
        )

        estimator = getattr(client, '_estimate_text_tokens', None)
        history_estimator = getattr(client, '_estimate_history_tokens', None)
        compression_enabled = bool(getattr(client, 'context_compression_enabled', False))
        upcoming_prompt_tokens = estimator(prompt) if estimator else len(prompt)
        existing_history_tokens = history_estimator() if history_estimator else 0
        context_limit_tokens = getattr(client, 'context_limit_tokens', None)
        safety_margin = max(512, int(context_limit_tokens * 0.1)) if context_limit_tokens else 0
        threshold_for_reconstruction = (
            max(0, (context_limit_tokens or 0) - safety_margin)
        ) if context_limit_tokens else None

        should_reconstruct = (
            context_limit_tokens is not None
            and keep_recent_entries > 0
            and already_processed
            and processed_translations
            and compression_enabled
            and (existing_history_tokens + upcoming_prompt_tokens >= (threshold_for_reconstruction or 0))
        )

        base_history: List[Dict[str, Any]] = []
        reconstruction_meta: Optional[Dict[str, Any]] = None
        summary_message: Optional[Dict[str, Any]] = None

        if should_reconstruct:
            compress_direct = getattr(client, '_compress_history_once', None)
            if callable(compress_direct):
                try:
                    compress_direct(0)
                except Exception as exc:
                    logger.warning("컨텍스트 요약 생성 실패: %s", exc)

            summary_message = _find_context_summary_message(client)
            if summary_message:
                logger.info("컨텍스트 요약 메시지가 재구성에 포함됩니다.")
            else:
                logger.warning("컨텍스트 요약 메시지를 찾지 못했습니다. 재구성 메시지만 사용합니다.")

            safety_margin = max(512, int(context_limit_tokens * 0.1)) if context_limit_tokens else 512
            threshold = max(0, (context_limit_tokens or 0) - safety_margin)
            candidate_keep = keep_recent_entries

            while candidate_keep >= 0:
                candidate_history, candidate_meta = _prepare_reconstructed_history(
                    client=client,
                    processed_subtitles=already_processed, # pyright: ignore[reportArgumentType]
                    translations=processed_translations,
                    keep_recent_entries=candidate_keep,
                    upcoming_prompt=prompt,
                )

                if not candidate_history:
                    if candidate_keep == 0:
                        break
                    candidate_keep -= 1
                    continue

                projected_tokens = candidate_meta['history_tokens'] + upcoming_prompt_tokens
                if context_limit_tokens is None or projected_tokens <= threshold or candidate_keep == 0:
                    base_history = candidate_history
                    reconstruction_meta = candidate_meta
                    if context_limit_tokens and projected_tokens > context_limit_tokens:
                        logger.warning(
                            "재구성 후에도 추정 토큰(%s)이 제한(%s)을 초과할 수 있습니다. batch_size를 줄이는 것을 고려하세요.",
                            projected_tokens,
                            context_limit_tokens,
                        )
                    break

                candidate_keep -= 1

            if base_history and reconstruction_meta:
                constructed_history: List[Dict[str, Any]] = []
                if summary_message:
                    constructed_history.append(summary_message)
                constructed_history.extend(base_history)
                base_history = constructed_history
                logger.info(
                    "재구성 히스토리 적용: 총 %s개 중 최근 %s개 유지, 재구성 토큰 %s, 모드=%s",
                    reconstruction_meta['total_entries'],
                    reconstruction_meta['recent_entries'],
                    reconstruction_meta['history_tokens'],
                    reconstruction_meta['mode'],
                )
            elif summary_message:
                base_history = [summary_message]
                logger.info("재구성 메시지 없이 컨텍스트 요약만 유지합니다.")

        retry_count = 0
        max_retries = 5
        batch_success = False
        confirmed_translations: Dict[int, str] = {}

        while retry_count < max_retries and not batch_success:
            # 중단 플래그 확인
            if stop_flag and stop_flag.get('stopped', False):
                logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
                return
                
            if base_history:
                client.start_chat(
                    history=_clone_history_messages(base_history),
                    suppress_log=True,
                    reset_usage=False,
                )
                
            try:
                buffer = ""
                processed_indices = set()
                batch_translations: Dict[int, str] = {}

                # response_schema를 사용하여 스트리밍 요청
                for chunk in client.send_message_stream(prompt, response_schema=response_schema):
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

                            if original_sub and item_index not in processed_indices:
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
                
                if len(processed_indices) == len(batch):
                    confirmed_translations = dict(batch_translations)
                    batch_success = True
                else:
                    raise ValueError("스트림 처리 중 일부 자막이 누락되었습니다.")

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.error(f"번역 재시도 {retry_count}회: {e}")
                    try:
                        client.delete_last_turn()
                    except Exception:
                        pass
                    time.sleep(10) 
                else:
                    error_message = f"배치 처리 실패 (재시도 {max_retries}회 초과): {e}"
                    confirmed_translations.update(batch_translations)
                    unprocessed_indices = set(original_sub_map.keys()) - processed_indices
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

@app.route('/')
def index():
    missing_api_key = not bool(os.environ.get('GOOGLE_API_KEY', '').strip())
    return render_template('index.html', missing_api_key=missing_api_key)

@app.route('/progress')
def progress():
    return render_template('progress.html')


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
    
    files_data = []
    for f in request.files.getlist('srt_files'):
        if f and f.filename:
            files_data.append({
                'name': f.filename,
                'text': f.read().decode('utf-8')
            })

    if not files_data:
        text = request.form.get('srt_text', '')
        if text.strip():
            files_data.append({'name': 'pasted.srt', 'text': text})

    if not files_data:
        return jsonify({'error': 'no srt data'}), 400

    job_id = str(uuid.uuid4())
    batch_size = request.form.get('batch_size', 10)
    try:
        batch_size = int(batch_size)
    except (ValueError, TypeError):
        batch_size = 10
    thinking_budget = request.form.get('thinking_budget', 8192)
    try:
        thinking_budget = int(thinking_budget)
    except (ValueError, TypeError):
        thinking_budget = 8192
    raw_context_compression = (request.form.get('context_compression') or '').strip()
    context_compression = 1 if raw_context_compression in {'1', 'true', 'True', 'on'} else 0
    context_limit = request.form.get('context_limit')
    try:
        context_limit = int(context_limit) if context_limit not in (None, '') else None
    except (ValueError, TypeError):
        context_limit = None
    api_key = (request.form.get('api_key') or '').strip()
    if api_key:
        # 사용자가 제출한 키를 즉시 .env에 저장하고 환경에 반영
        save_api_key_to_env(api_key)
    model_name = (request.form.get('model') or '').strip() or DEFAULT_MODEL
    job_data = {
        'files': files_data,
        'target_lang': request.form.get('target_lang', '한국어'),
        'batch_size': batch_size,
        'custom_prompt': request.form.get('custom_prompt', ''),
        'thinking_budget': thinking_budget,
        'context_compression': context_compression,
        'context_limit': context_limit,
        # 더 이상 DB에는 키를 저장하지 않습니다.
        'api_key': None,
        'model': model_name
    }
    save_job(job_id, job_data) # type: ignore
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
    thinking_budget = request.form.get('thinking_budget', 8192)
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
        logger.warning("thinking_budget 파라미터 변환 실패, 기본값 8192 사용: %s", thinking_budget)
        thinking_budget_val = 8192
    
    try:
        shared_client = GeminiClient(
            model=model_name,
            api_key=api_key if api_key else None,
            thinking_budget=thinking_budget_val,
            rpm_limit=int(get_config_value('rpm_limit', 9) or 9),
            generation_config={
                'max_output_tokens': 122880  # 120k 토큰
            },
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
    if thinking_budget_val <= 0:
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
                for line in translate_srt_stream(srt_content, shared_client, target_lang, batch_size, user_prompt, thinking_budget_val, stop_flag):
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

# ----- Preset management API (추가 설정과 독립) -----

@app.route('/api/presets', methods=['GET'])
def api_list_presets():
    """Return all stored presets."""
    return jsonify(list_presets())

@app.route('/api/presets/<name>', methods=['GET'])
def api_get_preset(name):
    preset = get_preset(name)
    if preset:
        # 추가 설정(API 키, 모델, thinking_budget, context 관련 등)은 프리셋과 분리됨.
        # (요청에 따라 thinking_budget은 프리셋에 포함해 저장/로드 가능하도록 허용)
        result = {
            'name': preset.get('name'),
            'target_lang': preset.get('target_lang'),
            'batch_size': preset.get('batch_size'),
            'custom_prompt': preset.get('custom_prompt'),
            'thinking_budget': preset.get('thinking_budget'),  # 0 또는 None 가능
        }
        return jsonify(result)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/presets/<name>', methods=['POST'])
def api_save_preset(name):
    data = request.get_json() or {}
    # 프리셋은 번역 기본 파라미터를 저장합니다.
    # (target_lang, batch_size, custom_prompt, thinking_budget)
    # thinking_budget은 요청에 따라 프리셋 단위 저장을 허용 (글로벌 기본값과 별개로 오버라이드 용도)
    target_lang = data.get('target_lang')
    batch_size_raw = data.get('batch_size')
    try:
        batch_size = int(batch_size_raw) if batch_size_raw is not None else None
    except (ValueError, TypeError):
        batch_size = None
    custom_prompt = data.get('custom_prompt')
    # thinking_budget 처리 (0 또는 양의 정수)
    thinking_budget_val = None
    if 'thinking_budget' in data:
        try:
            tb_raw = data.get('thinking_budget')
            if tb_raw is not None and tb_raw != '':
                tb_int = int(tb_raw)
                if tb_int >= 0:
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

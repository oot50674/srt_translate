import logging
import os
import tempfile
import json
import uuid
from typing import Optional, Dict, List, Any
from flask import Flask, render_template, request, jsonify, Response, send_file, abort
from dotenv import load_dotenv
from module import srt_module, ffmpeg_module
from module.gemini_module import GeminiClient
from module.subtitle_generation import (
    start_job as start_subtitle_job,
    get_job_data as get_subtitle_job,
    get_transcript_path as get_subtitle_transcript_path,
    get_segment_path as get_subtitle_segment_path,
)
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

# Import from separated modules
from constants import DEFAULT_MODEL, DEFAULT_CONTEXT_KEEP_RECENT, BASE_DIR
from utils import (
    save_api_key_to_env,
    save_history_log,
    translate_srt_stream,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# .env 로드 설정
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)


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


@app.route('/subtitle_jobs/<job_id>')
def subtitle_job_detail(job_id: str):
    """자막 생성 작업 모니터링 페이지."""
    job = get_subtitle_job(job_id)
    if not job:
        abort(404)
    return render_template('subtitle_job.html', job_id=job_id)


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


@app.route('/api/subtitle/jobs', methods=['POST'])
def api_create_subtitle_generation_job():
    youtube_url = (request.form.get('youtube_url') or '').strip()
    transcription_mode = (request.form.get('transcription_mode') or 'transcribe').strip() or 'transcribe'
    target_language = (request.form.get('target_language') or '').strip()
    custom_prompt = (request.form.get('custom_prompt') or '').strip()
    chunk_minutes_raw = request.form.get('chunk_minutes') or 10
    try:
        chunk_minutes = float(chunk_minutes_raw)
    except (TypeError, ValueError):
        chunk_minutes = 10.0
    video_file = request.files.get('video_file')
    srt_file = request.files.get('srt_file')
    try:
        job = start_subtitle_job(
            youtube_url=youtube_url or None,
            uploaded_file=video_file if video_file and video_file.filename else None,
            srt_file=srt_file if srt_file and srt_file.filename else None,
            chunk_minutes=chunk_minutes,
            mode=transcription_mode,
            target_language=target_language or None,
            custom_prompt=custom_prompt or None,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to create subtitle generation job")
        return jsonify({'error': '자막 생성 작업을 시작하지 못했습니다.'}), 500
    return jsonify({'job_id': job['job_id'], 'job': job})


@app.route('/api/subtitle/jobs/<job_id>', methods=['GET'])
def api_get_subtitle_generation_job(job_id: str):
    job = get_subtitle_job(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify(job)


@app.route('/api/subtitle/jobs/<job_id>/download/srt', methods=['GET'])
def api_download_subtitle_transcript(job_id: str):
    path = get_subtitle_transcript_path(job_id)
    if not path:
        return jsonify({'error': 'not ready'}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route('/api/subtitle/jobs/<job_id>/segments/<int:segment_index>/download', methods=['GET'])
def api_download_subtitle_segment(job_id: str, segment_index: int):
    path = get_subtitle_segment_path(job_id, segment_index)
    if not path:
        return jsonify({'error': 'not found'}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

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

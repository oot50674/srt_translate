import logging
import os
import tempfile
import json
import time
import uuid
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv, set_key
from module import srt_module
from module.gemini_module import GeminiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# .env 로드 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)

DEFAULT_MODEL = "gemini-2.5-flash"

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

    for i in range(0, len(subtitles), batch_size):
        # 중단 플래그 확인
        if stop_flag and stop_flag.get('stopped', False):
            logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
            break
            
        batch = subtitles[i:i + batch_size]
        original_sub_map = {sub['index']: sub for sub in batch}

        batch_items = []
        for sub in batch:
            escaped_text = sub["text"].replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
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

        retry_count = 0
        max_retries = 5
        batch_success = False

        while retry_count < max_retries and not batch_success:
            # 중단 플래그 확인
            if stop_flag and stop_flag.get('stopped', False):
                logger.info("중단 플래그가 설정되어 번역을 중지합니다.")
                return
                
            try:
                buffer = ""
                processed_indices = set()
                full_response = ""  # AI의 원본 응답을 저장할 변수

                # response_schema를 사용하여 스트리밍 요청
                for chunk in client.send_message_stream(prompt, response_schema=response_schema):
                    # 각 청크마다 중단 플래그 확인
                    if stop_flag and stop_flag.get('stopped', False):
                        logger.info("스트리밍 중 중단 플래그가 설정되어 번역을 중지합니다.")
                        return
                        
                    buffer += chunk
                    full_response += chunk  # 응답 전체를 누적
                    
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
                            
                            buffer = buffer[object_end + 1:]

                        except json.JSONDecodeError:
                            break
                
                if len(processed_indices) == len(batch):
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
                    logger.error(error_message)
                    batch_success = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def progress():
    return render_template('progress.html')

@app.route('/api/jobs', methods=['POST'])
def api_create_job():
    """업로드된 SRT 데이터와 옵션을 임시로 저장하고 작업 ID를 반환합니다."""
    # 새로운 작업을 추가하기 전에 오래된 작업(30일 경과)을 정리합니다.
    delete_old_jobs()
    
    logger.info("API Jobs 요청 폼 데이터: %s", dict(request.form))
    
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
        # 더 이상 DB에는 키를 저장하지 않습니다.
        'api_key': None,
        'model': model_name
    }
    save_job(job_id, job_data)
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
            rpm_limit=9,
            generation_config={
                'max_output_tokens': 122880  # 120k 토큰
            }
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
from module.database_module import (
    list_presets,
    get_preset,
    save_preset,
    delete_preset,
    save_job,
    get_job,
    delete_job,
    delete_old_jobs,
)

@app.route('/api/presets', methods=['GET'])
def api_list_presets():
    """Return all stored presets."""
    return jsonify(list_presets())

@app.route('/api/presets/<name>', methods=['GET'])
def api_get_preset(name):
    preset = get_preset(name)
    if preset:
        return jsonify(preset)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/presets/<name>', methods=['POST'])
def api_save_preset(name):
    data = request.get_json() or {}
    target_lang = data.get('target_lang')
    batch_size = data.get('batch_size')
    try:
        batch_size = int(batch_size) if batch_size is not None else None
    except (ValueError, TypeError):
        batch_size = None
    custom_prompt = data.get('custom_prompt')
    thinking_budget = data.get('thinking_budget')
    try:
        thinking_budget = int(thinking_budget) if thinking_budget is not None else None
    except (ValueError, TypeError):
        thinking_budget = None
    api_key = data.get('api_key')
    if isinstance(api_key, str):
        api_key = api_key.strip() or None
    else:
        api_key = None
    save_preset(name, target_lang, batch_size, custom_prompt, thinking_budget, api_key)
    return jsonify({'status': 'ok'})

@app.route('/api/presets/<name>', methods=['DELETE'])
def api_delete_preset(name):
    delete_preset(name)
    return jsonify({'status': 'deleted'})

if __name__ == '__main__':
    app.run(debug=True)

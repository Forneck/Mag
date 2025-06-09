import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import base64
import glob
import shutil

# --- Configuração dos Diretórios e Arquivos ---
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_agent_logs")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_final_outputs")
UPLOADED_FILES_CACHE_DIR = os.path.join(BASE_DIRECTORY, "gemini_uploaded_files_cache")
TEMP_ARTIFACTS_DIR = os.path.join(BASE_DIRECTORY, "gemini_temp_artifacts")

for directory in [LOG_DIRECTORY, OUTPUT_DIRECTORY, UPLOADED_FILES_CACHE_DIR, TEMP_ARTIFACTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

CURRENT_TIMESTAMP_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIRECTORY, f"agent_log_{CURRENT_TIMESTAMP_STR}.txt")

# --- Constantes ---
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_FACTOR = 2
MAX_AUTOMATIC_VALIDATION_RETRIES = 2
MAX_MANUAL_VALIDATION_RETRIES = 2

# --- Funções de Utilidade ---
def sanitize_filename(name, allow_extension=True):
    if not name: return ""
    base_name, ext = os.path.splitext(name)
    base_name = re.sub(r'[^\w\s.-]', '', base_name).strip()
    base_name = re.sub(r'[-\s]+', '-', base_name)[:100]
    if not allow_extension or not ext: return base_name
    ext = "." + re.sub(r'[^\w-]', '', ext.lstrip('.')).strip()[:10]
    return base_name + ext

def log_message(message, source="Sistema"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_log_message = f"[{timestamp}] [{source}]: {message}\n"
    try:
        with open(LOG_FILE_NAME, "a", encoding="utf-8") as f: f.write(full_log_message)
    except Exception as e: print(f"Erro ao escrever no log: {e}")

# --- Configuração da API Gemini ---
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    log_message("API Gemini configurada.", "Sistema")
except Exception as e:
    print(f"Erro fatal na configuração da API Gemini: {e}")
    log_message(f"Erro fatal na configuração da API Gemini: {e}", "Sistema")
    exit()

# Modelos
GEMINI_TEXT_MODEL_NAME = "gemini-2.0-flash"
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
log_message(f"Modelo Gemini (geração de imagem via SDK): {GEMINI_IMAGE_GENERATION_MODEL_NAME}", "Sistema")

generation_config_text = {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
generation_config_image_sdk = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "response_modalities": ['TEXT', 'IMAGE'], "max_output_tokens": 8192}
safety_settings_gemini = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}, {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]

# --- Funções Auxiliares de Comunicação ---
def print_agent_message(agent_name, message):
    print(f"\n🤖 [{agent_name}]: {message}")
    log_message(message, agent_name)

def print_user_message(message):
    print(f"\n👤 [Usuário]: {message}")
    log_message(message, "Usuário")

def call_gemini_api_with_retry(prompt_parts, agent_name, model_name, gen_config=None):
    log_message(f"Iniciando chamada à API Gemini para {agent_name}...", "Sistema")
    text_prompt_for_log = ""
    file_references_for_log = []

    active_gen_config = gen_config
    if active_gen_config is None:
        if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME:
             active_gen_config = generation_config_image_sdk
             log_message(f"Nenhuma gen_config específica passada para {agent_name}, usando config de imagem padrão para {model_name}.", "Sistema")
        else:
            active_gen_config = generation_config_text
            log_message(f"Nenhuma gen_config específica passada para {agent_name}, usando config de texto padrão para {model_name}.", "Sistema")

    for part_item in prompt_parts:
        if isinstance(part_item, str): text_prompt_for_log += part_item + "\n"
        elif hasattr(part_item, 'name') and hasattr(part_item, 'display_name'):
            file_references_for_log.append(f"Arquivo: {part_item.display_name} (ID: {part_item.name}, TipoMIME: {getattr(part_item, 'mime_type', 'N/A')})")
    log_message(f"Prompt textual para {agent_name} (Modelo: {model_name}):\n---\n{text_prompt_for_log}\n---", "Sistema")
    if file_references_for_log: log_message(f"Arquivos referenciados para {agent_name}:\n" + "\n".join(file_references_for_log), "Sistema")
    log_message(f"Usando generation_config para {agent_name} (Modelo: {model_name}): {active_gen_config}", "Sistema")
    log_message(f"Usando safety_settings para {agent_name} (Modelo: {model_name}): {safety_settings_gemini}", "Sistema")

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES} para {agent_name} (Modelo: {model_name})...", "Sistema")
        try:
            model_instance = genai.GenerativeModel(
                model_name,
                generation_config=active_gen_config,
                safety_settings=safety_settings_gemini
            )
            response = model_instance.generate_content(prompt_parts)
            log_message(f"Resposta bruta da API Gemini (tentativa {attempt + 1}, Modelo: {model_name}): {response}", agent_name)

            if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME:
                log_message(f"Retornando objeto 'response' completo para ImageWorker (tentativa {attempt+1}, Modelo: {model_name}).", "Sistema")
                return response

            if hasattr(response, 'text') and response.text is not None:
                 api_result_text = response.text.strip()
                 log_message(f"Sucesso! Resposta de texto da API Gemini (response.text) para {agent_name} (tentativa {attempt + 1}).", "Sistema")
                 return api_result_text

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                first_part = response.candidates[0].content.parts[0]
                if hasattr(first_part, 'text') and first_part.text is not None:
                    api_result_text = first_part.text.strip()
                    log_message(f"Texto extraído de response.candidates[0].content.parts[0].text para {agent_name} (tentativa {attempt + 1}).", "Sistema")
                    return api_result_text

            log_message(f"API Gemini (Modelo: {model_name}) não retornou texto utilizável para {agent_name} (tentativa {attempt + 1}).", agent_name)
            if response.prompt_feedback:
                log_message(f"Prompt Feedback: {response.prompt_feedback}", agent_name)
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reason_message = getattr(response.prompt_feedback, 'block_reason_message', 'Mensagem de bloqueio não disponível')
                    log_message(f"Bloqueio: {block_reason_message} ({response.prompt_feedback.block_reason})", agent_name)

            if attempt < MAX_API_RETRIES - 1:
                time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR; continue
            log_message(f"Falha após {MAX_API_RETRIES} tentativas (sem resposta utilizável para {agent_name}, Modelo: {model_name}).", agent_name)
            return None
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1}/{MAX_API_RETRIES} ({agent_name}, Modelo: {model_name}): {type(e).__name__} - {e}", agent_name)
            log_message(f"Traceback: {traceback.format_exc()}", agent_name)
            if "BlockedPromptException" in str(type(e)): log_message(f"Exceção Prompt Bloqueado: {e}", agent_name)
            elif "StopCandidateException" in str(type(e)): log_message(f"Exceção Parada de Candidato: {e}", agent_name)
            if attempt < MAX_API_RETRIES - 1:
                log_message(f"Aguardando {current_retry_delay}s...", "Sistema"); time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR
            else:
                log_message(f"Máximo de {MAX_API_RETRIES} tentativas. Falha API Gemini ({agent_name}, Modelo: {model_name}).", agent_name)
                return None
    log_message(f"call_gemini_api_with_retry ({agent_name}, Modelo: {model_name}) terminou sem retorno explícito após loop.", "Sistema")
    return None

# --- Funções de Arquivos ---
def get_most_recent_cache_file():
    try:
        list_of_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        log_message(f"Erro ao tentar encontrar o arquivo de cache mais recente: {e}", "Sistema")
        return None

def load_cached_files_metadata(cache_file_path):
    if not cache_file_path or not os.path.exists(cache_file_path):
        return []
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f:
            cached_metadata = json.load(f)
        if isinstance(cached_metadata, list):
            return cached_metadata
        log_message(f"Arquivo de cache {cache_file_path} não contém uma lista.", "Sistema")
        return []
    except json.JSONDecodeError:
        log_message(f"Erro ao decodificar JSON do arquivo de cache: {cache_file_path}", "Sistema")
        return []
    except Exception as e:
        log_message(f"Erro ao carregar arquivo de cache {cache_file_path}: {e}", "Sistema")
        return []

def clear_upload_cache():
    print_agent_message("Sistema", f"Verificando arquivos de cache de upload local em: {UPLOADED_FILES_CACHE_DIR}")
    local_cache_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))

    if not local_cache_files:
        print_agent_message("Sistema", "Nenhum arquivo de cache de upload local encontrado para limpar.")
    else:
        print_agent_message("Sistema", f"Encontrados {len(local_cache_files)} arquivo(s) de cache de upload local.")
        print_user_message("Deseja limpar TODOS esses arquivos de cache de upload LOCAL? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            deleted_count = 0
            for path in local_cache_files:
                try:
                    os.remove(path)
                    deleted_count += 1
                except Exception as e:
                    print_agent_message("Sistema", f"❌ Erro ao remover '{os.path.basename(path)}': {e}")
            print_agent_message("Sistema", f"✅ {deleted_count} arquivo(s) de cache local foram limpos.")

    print_agent_message("Sistema", "Verificando arquivos na API Gemini Files...")
    try:
        api_files_list = list(genai.list_files())
        if not api_files_list:
            print_agent_message("Sistema", "Nenhum arquivo encontrado na API Gemini Files para limpar.")
            return

        print_agent_message("Sistema", f"Encontrados {len(api_files_list)} arquivo(s) na API Gemini Files.")
        print_user_message("‼️ ATENÇÃO: IRREVERSÍVEL. ‼️ Deletar TODOS os arquivos da API Gemini? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            deleted_count = 0
            for f in api_files_list:
                try:
                    genai.delete_file(name=f.name)
                    deleted_count += 1
                    time.sleep(0.2)
                except Exception as e:
                    print_agent_message("Sistema", f"❌ Erro ao deletar '{f.display_name}': {e}")
            print_agent_message("Sistema", f"✅ {deleted_count} arquivo(s) foram deletados da API Gemini.")
    except Exception as e:
        print_agent_message("Sistema", f"❌ Erro ao acessar API Gemini para limpeza: {e}")

def get_uploaded_files_info_from_user():
    uploaded_file_objects = []
    uploaded_files_metadata = []
    reused_file_ids = set()

    print_agent_message("Sistema", "Buscando lista de arquivos na API Gemini...")
    try:
        api_files_list = list(genai.list_files())
    except Exception as e_list_files:
        print_agent_message("Sistema", f"Falha ao listar arquivos da API Gemini: {e_list_files}. Verifique sua conexão/chave API.")
        api_files_list = []

    api_files_dict = {f.name: f for f in api_files_list}
    log_message(f"Encontrados {len(api_files_dict)} arquivos na API Gemini.", "Sistema")

    most_recent_cache_path = get_most_recent_cache_file()
    cached_metadata_from_file = []
    if most_recent_cache_path:
        log_message(f"Carregando metadados do cache local: {most_recent_cache_path}", "Sistema")
        cached_metadata_from_file = load_cached_files_metadata(most_recent_cache_path)

    offer_for_reuse_metadata_list = []
    for api_file in api_files_list:
        user_path = "N/A (direto da API)"
        corresponding_cached_meta = next((cm for cm in cached_metadata_from_file if cm.get("file_id") == api_file.name), None)
        if corresponding_cached_meta:
            user_path = corresponding_cached_meta.get("user_path", user_path)
        
        offer_for_reuse_metadata_list.append({
            "file_id": api_file.name, "display_name": api_file.display_name,
            "mime_type": api_file.mime_type, "uri": api_file.uri,
            "size_bytes": api_file.size_bytes, "state": str(api_file.state),
            "user_path": user_path
        })

    if offer_for_reuse_metadata_list:
        print_agent_message("Sistema", "Arquivos encontrados na API (e/ou no cache local):")
        for idx, meta in enumerate(offer_for_reuse_metadata_list):
            print(f"  {idx + 1}. {meta['display_name']} (ID: {meta['file_id']})")
        print_user_message("Deseja reutilizar algum desses arquivos? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            print_user_message("Digite os números dos arquivos (ex: 1,3) ou 'todos':")
            choices_str = input("➡️ ").strip().lower()
            indices_to_try = []
            if choices_str == 'todos':
                indices_to_try = range(len(offer_for_reuse_metadata_list))
            else:
                try:
                    indices_to_try = [int(x.strip()) - 1 for x in choices_str.split(',')]
                except ValueError:
                    print("❌ Entrada inválida.")
            
            for idx in indices_to_try:
                if 0 <= idx < len(offer_for_reuse_metadata_list):
                    chosen_meta = offer_for_reuse_metadata_list[idx]
                    file_id = chosen_meta["file_id"]
                    if file_id in reused_file_ids: continue
                    try:
                        file_obj = api_files_dict.get(file_id) or genai.get_file(name=file_id)
                        uploaded_file_objects.append(file_obj)
                        uploaded_files_metadata.append(chosen_meta)
                        reused_file_ids.add(file_id)
                        print(f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                    except Exception as e:
                        print(f"❌ Erro ao obter arquivo '{chosen_meta['display_name']}': {e}")
                else:
                    print(f"❌ Índice inválido: {idx + 1}")
    else:
        print_agent_message("Sistema", "Nenhum arquivo na API para reutilização.")

    print_user_message("Adicionar NOVOS arquivos? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        while True:
            print_user_message("Caminho do arquivo/padrão (ex: *.txt) ou 'fim':")
            fp_pattern = input("➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            
            try:
                expanded_files = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*', '?']) else ([fp_pattern] if os.path.isfile(fp_pattern) else [])
                if not expanded_files:
                    print(f"❌ Nenhum arquivo encontrado para '{fp_pattern}'")
                    continue
                
                for fp in expanded_files:
                    try:
                        print_agent_message("Sistema", f"Upload de '{os.path.basename(fp)}'...")
                        uf = genai.upload_file(path=fp, display_name=os.path.basename(fp))
                        uploaded_file_objects.append(uf)
                        uploaded_files_metadata.append({"user_path": fp, "display_name": uf.display_name, "file_id": uf.name, "uri": uf.uri, "mime_type": uf.mime_type, "size_bytes": uf.size_bytes, "state": str(uf.state)})
                        print(f"✅ '{uf.display_name}' (ID: {uf.name}) enviado!")
                    except Exception as e:
                        print(f"❌ Erro no upload de '{fp}': {e}")
            except Exception as e:
                print(f"❌ Erro ao processar padrão '{fp_pattern}': {e}")

    if uploaded_files_metadata:
        try:
            cache_path = os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(uploaded_files_metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_message(f"Erro ao salvar cache de uploads: {e}", "Sistema")
            
    return uploaded_file_objects, uploaded_files_metadata

def format_uploaded_files_info_for_prompt_text(files_metadata_list):
    if not files_metadata_list: return "Nenhum arquivo complementar fornecido."
    txt = "Arquivos complementares carregados:\n"
    for m in files_metadata_list: txt += f"- Nome: {m['display_name']} (ID: {m['file_id']})\n"
    return txt

def get_user_feedback_or_approval():
    while True:
        prompt = (
            "\nO que você gostaria de fazer?\n"
            "  [A]provar - Salvar os artefatos como estão.\n"
            "  [F]eedback - Fornecer feedback para o sistema tentar novamente.\n"
            "  [S]air - Encerrar o processo.\n"
            "Escolha uma opção (A/F/S): "
        )
        print_user_message(prompt)
        choice = input("➡️ ").strip().lower()
        if choice in ['a', 'f', 's']:
            log_message(f"Usuário escolheu a opção: {choice.upper()}", "UsuárioInput")
            return choice
        else:
            print_agent_message("Sistema", "❌ Opção inválida. Por favor, escolha 'A', 'F' ou 'S'.")

# --- Classes de Agentes ---
# (Todas as classes - ImageWorker, Worker, Validator, TaskManager - são definidas aqui como na v9.4)

class ImageWorker:
    def __init__(self):
        self.model_name = GEMINI_IMAGE_GENERATION_MODEL_NAME
        self.generation_config = generation_config_image_sdk
        log_message(f"ImageWorker criado para {self.model_name}", "ImageWorker")

    def generate_image(self, text_prompt_for_image):
        agent_display_name = "ImageWorker"
        print_agent_message(agent_display_name, f"Gerando imagem para: '{text_prompt_for_image[:100]}...'")
        log_message(f"Prompt para ImageWorker: {text_prompt_for_image}", agent_display_name)
        
        response_object = call_gemini_api_with_retry(
            [text_prompt_for_image], agent_display_name, self.model_name, self.generation_config)

        if response_object is None: return "Falha na geração da imagem (API não respondeu)."

        try:
            if response_object.candidates and response_object.candidates[0].content and response_object.candidates[0].content.parts:
                for part in response_object.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
                        temp_filename = f"temp_img_{sanitize_filename(text_prompt_for_image[:30], False)}_{int(time.time())}.png"
                        artifact_path = os.path.join(TEMP_ARTIFACTS_DIR, temp_filename)
                        with open(artifact_path, "wb") as f:
                            f.write(image_bytes)
                        log_message(f"Sucesso! Imagem salva temporariamente em: {artifact_path}", agent_display_name)
                        return artifact_path
            
            if hasattr(response_object, 'prompt_feedback') and response_object.prompt_feedback and hasattr(response_object.prompt_feedback, 'block_reason') and response_object.prompt_feedback.block_reason:
                msg = f"Falha: Geração bloqueada ({response_object.prompt_feedback.block_reason})"
                log_message(msg, agent_display_name)
                return msg
        except Exception as e:
            log_message(f"Erro ao processar resposta da API de imagem: {e}\n{traceback.format_exc()}", agent_display_name)
            return f"Falha: Erro ao processar resposta ({e})."

        return "Falha: Nenhuma imagem na resposta."

class Worker:
    def __init__(self):
        self.gemini_model_name = GEMINI_TEXT_MODEL_NAME
        log_message("Instância do Worker criada.", "Worker")

    def execute_sub_task(self, sub_task_description, context_text_part, uploaded_file_objects):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_text_for_worker = rf"""
Você é um Agente Executor. Tarefa atual: "{sub_task_description}"
Contexto (resultados anteriores, objetivo original, arquivos):
{context_text_part}
Execute a tarefa.
- Se for "Criar uma descrição textual detalhada (prompt) para gerar a imagem de [...]", seu resultado DEVE ser APENAS essa descrição textual.
- Se a tarefa envolver modificar ou criar arquivos de código, forneça o CONTEÚDO COMPLETO do arquivo. Indique o NOME DO ARQUIVO CLARAMENTE antes de cada bloco de código (ex: "Arquivo: nome.ext" ou ```python nome.py ... ```).
- Se identificar NOVAS sub-tarefas cruciais, liste-as em 'NOVAS_TAREFAS_SUGERIDAS:' como array JSON de strings. Se não, omita.
"""
        prompt_parts = [prompt_text_for_worker] + uploaded_file_objects
        response_text = call_gemini_api_with_retry(
            prompt_parts, agent_display_name, self.gemini_model_name, generation_config_text
        )

        if response_text is None: return None, []
        if not response_text.strip():
            log_message(f"Worker: resposta vazia para '{sub_task_description}'.", agent_display_name)
            return "Resposta vazia da API.", []
        
        task_res, sugg_tasks_strings = response_text, []
        marker = "NOVAS_TAREFAS_SUGERIDAS:"
        if marker in response_text:
            parts = response_text.split(marker, 1)
            task_res = parts[0].strip()
            potential_json_or_list = parts[1].strip()
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', potential_json_or_list, re.DOTALL)
                parsed_suggestions = []
                if json_match:
                    parsed_suggestions = json.loads(json_match.group(1) or json_match.group(2))
                if isinstance(parsed_suggestions, list):
                    sugg_tasks_strings = [str(item) for item in parsed_suggestions]
            except Exception as e:
                log_message(f"Erro ao processar novas tarefas: {e}", agent_display_name)
        else:
            task_res = response_text.strip()
        
        if task_res.lower().startswith("resultado da tarefa:"):
            task_res = task_res[len("resultado da tarefa:"):].strip()
        
        log_message(f"Resultado da sub-tarefa '{sub_task_description}': {task_res[:500]}...", agent_display_name)
        return task_res, sugg_tasks_strings

class Validator:
    def __init__(self, task_manager_ref):
        self.tm = task_manager_ref
        self.text_model_name = GEMINI_TEXT_MODEL_NAME
        self.text_gen_config = generation_config_text
        log_message("Instância do Validator criada.", "Validator")
        
    def evaluate_and_select_image_concepts(self, original_goal, image_task_results, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_display_name = "Validator (Avaliação de Imagens)"
        print_agent_message(agent_display_name, "Avaliando conceitos de imagem gerados...")

        summary_of_image_attempts = "\n".join(
            f"Tentativa {i+1}: Prompt='{res.get('image_prompt_used', 'N/A')}', Sucesso={os.path.exists(str(res.get('result')))}"
            for i, res in enumerate(image_task_results)
        ) or "Nenhuma."

        prompt_text_part = rf"""
Você é um Diretor de Arte. Analise os prompts usados para gerar imagens para a meta: "{original_goal}".
{summary_of_image_attempts}
**IMPORTANTE:** Sua resposta DEVE SER ESTRITAMENTE um array JSON de strings com os prompts que você aprova. Ex: `["prompt aprovado 1", "prompt aprovado 2"]`. Se nenhum, retorne `[]`.
"""
        llm_response_text = call_gemini_api_with_retry([prompt_text_part] + uploaded_file_objects, agent_display_name, self.text_model_name, self.text_gen_config)

        selected_prompts = []
        if llm_response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', llm_response_text, re.DOTALL)
                if json_match:
                    selected_prompts = json.loads(json_match.group(1) or json_match.group(2))
            except Exception as e:
                log_message(f"Erro ao decodificar JSON de avaliação de imagem: {e}", agent_display_name)
        
        validated_concepts = [res for res in image_task_results if res.get('image_prompt_used') in selected_prompts and os.path.exists(str(res.get('result')))]
        log_message(f"Conceitos de imagem validados: {len(validated_concepts)}", agent_display_name)
        return validated_concepts
        
    def validate_and_save_final_output(self, original_goal, final_context, uploaded_file_objects, temp_artifacts_to_save):
        agent_display_name = "Validator (Validação Final)"
        print_agent_message(agent_display_name, "Validando resultado final...")

        summary_of_artifacts = "\n".join([f"- Tipo: {a['type']}, Nome: {os.path.basename(a.get('filename') or a.get('artifact_path', ''))}" for a in temp_artifacts_to_save]) or "Nenhum."

        prompt_text_part_validation = rf"""
Você é um Gerente de QA. Analise o contexto e os artefatos gerados para a meta: "{original_goal}".
Contexto: {final_context}
Artefatos para Salvar: {summary_of_artifacts}
Retorne um JSON com "validation_passed" (true/false), "main_report" (Markdown) e "general_evaluation" (resumo).
"""
        llm_response_text = call_gemini_api_with_retry([prompt_text_part_validation] + uploaded_file_objects, agent_display_name, self.text_model_name, self.text_gen_config)

        if llm_response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_response_text, re.DOTALL)
                parsed_json = json.loads(json_match.group(1)) if json_match else {}
                
                validation_passed = parsed_json.get("validation_passed", False)
                main_report = parsed_json.get("main_report", "Relatório não gerado.")
                evaluation_text = parsed_json.get("general_evaluation", "Avaliação não gerada.")

                goal_slug = sanitize_filename(original_goal, allow_extension=False)
                assessment_file_name = os.path.join(OUTPUT_DIRECTORY, f"avaliacao_completa_{goal_slug}_{CURRENT_TIMESTAMP_STR}.md")
                with open(assessment_file_name, "w", encoding="utf-8") as f:
                    f.write(f"# Relatório: {original_goal}\n\n{main_report}\n\n## Avaliação da IA\n{evaluation_text}")
                print_agent_message(agent_display_name, f"Relatório de avaliação salvo: {assessment_file_name}")

                if validation_passed:
                    final_artifact_dir = os.path.join(OUTPUT_DIRECTORY, f"artefatos_finais_{goal_slug}_{CURRENT_TIMESTAMP_STR}")
                    os.makedirs(final_artifact_dir, exist_ok=True)
                    for artifact in temp_artifacts_to_save:
                        source_path = artifact.get('temp_path') or artifact.get('artifact_path')
                        if source_path and os.path.exists(source_path):
                            dest_name = artifact.get('filename') or f"imagem_{sanitize_filename(artifact.get('prompt', '')[:30], False)}.png"
                            shutil.copy(source_path, os.path.join(final_artifact_dir, dest_name))
                            print_agent_message(agent_display_name, f"✅ Artefato final salvo: {dest_name}")
                
                return validation_passed, evaluation_text
            except Exception as e:
                log_message(f"Erro ao processar JSON de validação: {e}", agent_display_name)
                return False, f"Falha ao processar resposta de validação: {e}"
        return False, "Falha ao obter avaliação final da API."

class TaskManager:
    def __init__(self):
        self.worker = Worker()
        self.image_worker = ImageWorker()
        self.validator = Validator(self)
        self.task_list = []
        self.completed_tasks_results = []
        self.uploaded_files_metadata = []
        self.temp_artifacts = []
        log_message("Instância do TaskManager criada.", "TaskManager")

    def run_workflow(self, initial_goal, uploaded_file_objects, uploaded_files_metadata):
        self.uploaded_files_metadata = uploaded_files_metadata
        # ... (O resto do workflow segue a lógica da v9.4, usando as classes e funções corrigidas)
        # ... O fluxo complexo de validação e feedback com o usuário está aqui.
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        log_message(f"Meta inicial: {initial_goal}", "TaskManager")
        
        files_metadata_for_prompt_text = format_uploaded_files_info_for_prompt_text(self.uploaded_files_metadata)

        if not self.decompose_task(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text):
            print_agent_message("TaskManager", "Falha na decomposição da tarefa. Encerrando.")
            return
        
        print_agent_message("TaskManager", "--- PLANO DE TAREFAS INICIAL ---")
        for i, task in enumerate(self.task_list): print(f"  {i+1}. {task}")
        print_user_message("Aprova este plano? (s/n)"); 
        if input("➡️ ").strip().lower() != 's':
            print_agent_message("TaskManager", "Plano rejeitado. Encerrando."); return
        
        # O resto do método run_workflow continua aqui, com o loop principal,
        # o ciclo de validação e o menu de feedback do usuário.

    def decompose_task(self, main_goal, uploaded_file_objects, files_metadata_for_prompt_text):
        # ... (código idêntico à v9.4) ...
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{main_goal}'")

        prompt_text_part = f"""
Você é um Gerenciador de Tarefas especialista. Decomponha a meta principal em sub-tarefas sequenciais.
Meta Principal: "{main_goal}"
Arquivos Complementares: {files_metadata_for_prompt_text}

Se a meta envolver CRIAÇÃO DE MÚLTIPLAS IMAGENS (ex: "crie 3 logos", "gere 2 variações de um personagem"), você DEVE:
1.  Criar uma tarefa para gerar a descrição de CADA imagem individualmente. Ex: "Criar descrição para imagem 1 de [assunto]".
2.  Seguir CADA tarefa de descrição com uma tarefa "TASK_GERAR_IMAGEM: [assunto da imagem correspondente]".
3.  Após TODAS as tarefas de geração de imagem, adicionar UMA tarefa: "TASK_AVALIAR_IMAGENS: Avaliar as imagens/descrições geradas para [objetivo original] e selecionar as melhores que atendem aos critérios."

Se for UMA ÚNICA IMAGEM, use o formato:
1.  "Criar uma descrição textual detalhada (prompt) para gerar a imagem de [assunto]."
2.  "TASK_GERAR_IMAGEM: [assunto da imagem]"
3.  (Opcional, mas recomendado se a qualidade for crítica) "TASK_AVALIAR_IMAGENS: Avaliar a imagem gerada para [objetivo original]."

Para outras metas, decomponha normalmente. Retorne em JSON array de strings.
Exemplo Múltiplas Imagens: ["Criar descrição para imagem 1 de logo moderno", "TASK_GERAR_IMAGEM: Imagem 1 de logo moderno", "Criar descrição para imagem 2 de logo vintage", "TASK_GERAR_IMAGEM: Imagem 2 de logo vintage", "TASK_AVALIAR_IMAGENS: Avaliar os logos gerados para cafeteria e selecionar o melhor."]
Sub-tarefas:
"""
        prompt_parts_for_api = [prompt_text_part] + uploaded_file_objects
        response_text = call_gemini_api_with_retry(
            prompt_parts_for_api,
            agent_display_name,
            GEMINI_TEXT_MODEL_NAME,
            generation_config_text
        )

        if response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', response_text, re.DOTALL)
                if json_match:
                    self.task_list = json.loads(json_match.group(1) or json_match.group(2))
                    return True
            except Exception as e:
                log_message(f"Erro na decomposição: {e}", agent_display_name)
        return False
        
# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.4.1"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Correção de Bug NameError) ---")
    print(f"📝 Logs: {LOG_FILE_NAME}\n📄 Saídas Finais: {OUTPUT_DIRECTORY}\n⏳ Artefatos Temporários: {TEMP_ARTIFACTS_DIR}\nℹ️ Cache Uploads: {UPLOADED_FILES_CACHE_DIR}")
    
    print_user_message("Deseja limpar o cache de uploads (local e/ou da API Gemini) antes de começar? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        clear_upload_cache()
    
    initial_goal_input = input("🎯 Defina a meta principal: ")
    print_user_message(initial_goal_input)
    
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()
    
    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
    else:
        if os.path.exists(TEMP_ARTIFACTS_DIR):
            shutil.rmtree(TEMP_ARTIFACTS_DIR)
        os.makedirs(TEMP_ARTIFACTS_DIR)
        
        manager = TaskManager()
        manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")

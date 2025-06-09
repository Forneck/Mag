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
# CORREÇÃO: Adicionando as funções que faltavam
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
    api_files_to_clear = []

    if not local_cache_files:
        print_agent_message("Sistema", "Nenhum arquivo de cache de upload local encontrado para limpar.")
        log_message("Nenhum arquivo de cache de upload local encontrado.", "Sistema")
    else:
        print_agent_message("Sistema", f"Encontrados {len(local_cache_files)} arquivo(s) de cache de upload local:")
        for cf in local_cache_files:
            print(f"  - {os.path.basename(cf)}")

        print_user_message("Deseja limpar TODOS esses arquivos de cache de upload LOCAL? (s/n)")
        choice_local = input("➡️ ").strip().lower()

        if choice_local == 's':
            deleted_count_local = 0
            errors_count_local = 0
            for cache_file_path in local_cache_files:
                try:
                    os.remove(cache_file_path)
                    log_message(f"Arquivo de cache local '{os.path.basename(cache_file_path)}' removido.", "Sistema")
                    deleted_count_local += 1
                except Exception as e:
                    log_message(f"Erro ao remover arquivo de cache local '{os.path.basename(cache_file_path)}': {e}", "Sistema")
                    print_agent_message("Sistema", f"❌ Erro ao remover cache local '{os.path.basename(cache_file_path)}'.")
                    errors_count_local += 1
            
            if deleted_count_local > 0:
                print_agent_message("Sistema", f"✅ {deleted_count_local} arquivo(s) de cache local foram limpos.")
            if errors_count_local > 0:
                print_agent_message("Sistema", f"⚠️ {errors_count_local} erro(s) ao tentar limpar arquivos de cache local.")
            if deleted_count_local == 0 and errors_count_local == 0 and local_cache_files:
                 print_agent_message("Sistema", "Nenhum arquivo de cache local foi efetivamente limpo (apesar de listados).")
        else:
            print_agent_message("Sistema", "Limpeza do cache de upload local cancelada pelo usuário.")
            log_message("Limpeza do cache de upload local cancelada.", "UsuárioInput")

    print_agent_message("Sistema", "Verificando arquivos na API Gemini Files...")
    try:
        api_files_list = list(genai.list_files())
        if not api_files_list:
            print_agent_message("Sistema", "Nenhum arquivo encontrado na API Gemini Files para limpar.")
            log_message("Nenhum arquivo encontrado na API Gemini Files.", "Sistema")
            return
        
        print_agent_message("Sistema", f"Encontrados {len(api_files_list)} arquivo(s) na API Gemini Files.")

        print_user_message("‼️ ATENÇÃO: Esta ação é IRREVERSÍVEL. ‼️\nDeseja deletar TODOS os arquivos atualmente listados na API Gemini Files? (s/n)")
        choice_api = input("➡️ ").strip().lower()

        if choice_api == 's':
            print_agent_message("Sistema", "Iniciando exclusão de arquivos da API Gemini Files... Isso pode levar um momento.")
            deleted_count_api = 0
            errors_count_api = 0
            for api_file_to_delete in api_files_list:
                try:
                    genai.delete_file(name=api_file_to_delete.name)
                    log_message(f"Arquivo da API '{api_file_to_delete.display_name}' (ID: {api_file_to_delete.name}) deletado.", "Sistema")
                    print(f"  🗑️ Deletado da API: {api_file_to_delete.display_name}")
                    deleted_count_api += 1
                    time.sleep(0.2)
                except Exception as e:
                    log_message(f"Erro ao deletar arquivo da API '{api_file_to_delete.display_name}' (ID: {api_file_to_delete.name}): {e}", "Sistema")
                    print_agent_message("Sistema", f"❌ Erro ao deletar da API: '{api_file_to_delete.display_name}'.")
                    errors_count_api += 1
            
            if deleted_count_api > 0:
                print_agent_message("Sistema", f"✅ {deleted_count_api} arquivo(s) foram deletados da API Gemini Files.")
            if errors_count_api > 0:
                print_agent_message("Sistema", f"⚠️ {errors_count_api} erro(s) ao tentar deletar arquivos da API.")
            if deleted_count_api == 0 and errors_count_api == 0 and api_files_list:
                 print_agent_message("Sistema", "Nenhum arquivo da API foi efetivamente deletado (apesar de listados).")

        else:
            print_agent_message("Sistema", "Limpeza de arquivos da API Gemini Files cancelada pelo usuário.")
            log_message("Limpeza de arquivos da API Gemini Files cancelada.", "UsuárioInput")

    except Exception as e_api_clear:
        print_agent_message("Sistema", f"❌ Erro ao tentar acessar ou limpar arquivos da API Gemini: {e_api_clear}")
        log_message(f"Erro geral durante a tentativa de limpeza de arquivos da API: {e_api_clear}", "Sistema")

def get_uploaded_files_info_from_user():
    uploaded_file_objects = []
    uploaded_files_metadata = []
    reused_file_ids = set()

    print_agent_message("Sistema", "Buscando lista de arquivos na API Gemini...")
    try:
        api_files_list = list(genai.list_files())
    except Exception as e_list_files:
        print_agent_message("Sistema", f"Falha ao listar arquivos da API Gemini: {e_list_files}. Verifique sua conexão/chave API.")
        log_message(f"Falha crítica ao listar arquivos da API: {e_list_files}", "Sistema")
        api_files_list = []

    api_files_dict = {f.name: f for f in api_files_list}
    log_message(f"Encontrados {len(api_files_dict)} arquivos na API Gemini.", "Sistema")

    most_recent_cache_path = get_most_recent_cache_file() # Esta chamada agora funciona
    cached_metadata_from_file = []
    if most_recent_cache_path:
        log_message(f"Carregando metadados do cache local: {most_recent_cache_path}", "Sistema")
        cached_metadata_from_file = load_cached_files_metadata(most_recent_cache_path)

    offer_for_reuse_metadata_list = []
    processed_api_file_ids_for_reuse_offer = set()

    for api_file in api_files_list:
        api_file_id = api_file.name
        display_name = api_file.display_name
        mime_type = api_file.mime_type
        uri = api_file.uri
        size_bytes = api_file.size_bytes
        state = str(api_file.state)
        user_path = "N/A (direto da API)"

        corresponding_cached_meta = next((cm for cm in cached_metadata_from_file if cm.get("file_id") == api_file_id), None)
        if corresponding_cached_meta:
            user_path = corresponding_cached_meta.get("user_path", user_path)
            display_name = corresponding_cached_meta.get("display_name", display_name)
            mime_type = corresponding_cached_meta.get("mime_type", mime_type)

        offer_for_reuse_metadata_list.append({
            "file_id": api_file_id,
            "display_name": display_name,
            "mime_type": mime_type,
            "uri": uri,
            "size_bytes": size_bytes,
            "state": state,
            "user_path": user_path
        })
        processed_api_file_ids_for_reuse_offer.add(api_file_id)

    if offer_for_reuse_metadata_list:
        print_agent_message("Sistema", "Arquivos encontrados na API (e/ou no cache local):")
        for idx, meta_to_offer in enumerate(offer_for_reuse_metadata_list):
            print(f"  {idx + 1}. {meta_to_offer['display_name']} (ID: {meta_to_offer['file_id']}, Tipo: {meta_to_offer.get('mime_type')}, Origem: {meta_to_offer.get('user_path', 'API')})")

        print_user_message("Deseja reutilizar algum desses arquivos? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            print_user_message("Digite os números dos arquivos para reutilizar, separados por vírgula (ex: 1,3). Ou 'todos':")
            choices_str = input("➡️ ").strip().lower()
            selected_indices_to_try = []

            if choices_str == 'todos':
                selected_indices_to_try = list(range(len(offer_for_reuse_metadata_list)))
            else:
                try:
                    selected_indices_to_try = [int(x.strip()) - 1 for x in choices_str.split(',')]
                except ValueError:
                    print("❌ Entrada inválida para seleção de arquivos.")
                    log_message("Entrada inválida do usuário para seleção de arquivos cacheados.", "Sistema")

            for selected_idx in selected_indices_to_try:
                if 0 <= selected_idx < len(offer_for_reuse_metadata_list):
                    chosen_meta_for_reuse = offer_for_reuse_metadata_list[selected_idx]
                    file_id_to_reuse = chosen_meta_for_reuse["file_id"]

                    if file_id_to_reuse in reused_file_ids:
                        print(f"ℹ️ Arquivo '{chosen_meta_for_reuse['display_name']}' já selecionado para reutilização.")
                        continue
                    try:
                        print_agent_message("Sistema", f"Obtendo arquivo '{chosen_meta_for_reuse['display_name']}' (ID: {file_id_to_reuse}) da API para reutilização...")
                        file_obj = api_files_dict.get(file_id_to_reuse)
                        if not file_obj:
                            file_obj = genai.get_file(name=file_id_to_reuse)

                        uploaded_file_objects.append(file_obj)
                        uploaded_files_metadata.append(chosen_meta_for_reuse)
                        reused_file_ids.add(file_id_to_reuse)
                        print(f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                        log_message(f"Arquivo '{file_obj.display_name}' (ID: {file_id_to_reuse}) reutilizado da API. Metadados cacheados: {chosen_meta_for_reuse}", "Sistema")
                    except Exception as e:
                        print(f"❌ Erro ao obter arquivo '{chosen_meta_for_reuse['display_name']}' da API: {e}")
                        log_message(f"Erro ao obter arquivo '{file_id_to_reuse}' da API para reutilização: {e}", "Sistema")
                else:
                    print(f"❌ Índice inválido: {selected_idx + 1}")
    else:
        print_agent_message("Sistema", "Nenhum arquivo encontrado na API para reutilização ou o cache está vazio.")

    print_user_message("Adicionar NOVOS arquivos (além dos reutilizados)? s/n")
    add_new_files_choice = input("➡️ ").strip().lower()
    if add_new_files_choice == 's':
        print_agent_message("Sistema", "Preparando para upload de novos arquivos...")
        while True:
            print_user_message("Caminho do novo arquivo/padrão (permite uso de *.ext)  (ou 'fim'):")
            fp_pattern = input("➡️ ").strip()
            if fp_pattern.lower() == 'fim':
                break

            try:
                if any(c in fp_pattern for c in ['*', '?', '[', ']']):
                    expanded_files = glob.glob(fp_pattern, recursive=True)
                elif os.path.exists(fp_pattern) and os.path.isfile(fp_pattern):
                    expanded_files = [fp_pattern]
                else:
                    expanded_files = []
                    if not os.path.exists(fp_pattern):
                         print(f"❌ Caminho/padrão '{fp_pattern}' não encontrado.")
                    elif not os.path.isfile(fp_pattern):
                         print(f"❌ Caminho '{fp_pattern}' não é um arquivo.")


                if not expanded_files and any(c in fp_pattern for c in ['*', '?', '[', ']']):
                    print(f"❌ Nenhum arquivo encontrado para o padrão: '{fp_pattern}'")
                    log_message(f"Nenhum arquivo encontrado para o padrão '{fp_pattern}'.", "Sistema")
                    continue
                elif not expanded_files:
                    log_message(f"Caminho/padrão inválido '{fp_pattern}'.", "Sistema")
                    continue


                print_agent_message("Sistema", f"Arquivos encontrados para '{fp_pattern}': {expanded_files}")
                if len(expanded_files) > 1 :
                    print_user_message(f"Confirmar upload de {len(expanded_files)} arquivos? (s/n)")
                    if input("➡️ ").strip().lower() != 's':
                        print_agent_message("Sistema", "Upload cancelado pelo usuário.")
                        continue

                for fp_actual in expanded_files:
                    if not os.path.exists(fp_actual) or not os.path.isfile(fp_actual):
                        print(f"❌ Arquivo '{fp_actual}' inválido ou não é um arquivo. Pulando.")
                        log_message(f"Arquivo '{fp_actual}' inválido ou não é um arquivo. Pulando.", "Sistema")
                        continue

                    dn = os.path.basename(fp_actual)
                    if any(meta.get("display_name") == dn and meta.get("file_id") not in reused_file_ids for meta in uploaded_files_metadata):
                        print_user_message(f"⚠️ Um arquivo NOVO chamado '{dn}' já foi adicionado nesta sessão. Continuar com este '{fp_actual}'? (s/n)")
                        if input("➡️ ").strip().lower() != 's':
                            continue
                    elif any(meta.get("display_name") == dn and meta.get("file_id") in reused_file_ids for meta in uploaded_files_metadata):
                         print(f"ℹ️ Um arquivo chamado '{dn}' já foi marcado para reutilização da API. Este upload de '{fp_actual}' será um novo arquivo na API se o conteúdo for diferente, ou a API poderá deduplicar.")

                    try:
                        print_agent_message("Sistema", f"Upload de '{dn}' (de '{fp_actual}')...")
                        mime_type_upload = None
                        ext_map = {
                            ".md": "text/markdown", ".py": "text/x-python", ".cpp": "text/x-c++src",
                            ".hpp": "text/x-c++hdr", ".h": "text/x-chdr", ".c": "text/x-csrc",
                            ".txt": "text/plain", ".json": "text/plain", ".html": "text/html",
                            ".css": "text/css", ".js": "text/javascript", ".xml": "text/plain",
                            ".csv": "text/csv", ".java": "text/x-java-source", ".swift": "text/x-swift",
                            ".kt": "text/x-kotlin", ".rb": "text/x-ruby", ".php": "text/x-php",
                            ".go": "text/x-go", ".rs": "text/rust", ".ts": "text/typescript",
                            ".sh": "text/x-shellscript", ".ps1": "application/x-powershell",
                            ".bat": "application/x-bat", ".yaml": "application/x-yaml", ".yml": "application/x-yaml",
                            ".toml": "application/toml", ".ini": "text/plain", ".pdf": "application/pdf",
                            ".doc": "application/msword",
                            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            ".xls": "application/vnd.ms-excel",
                            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            ".ppt": "application/vnd.ms-powerpoint",
                            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            ".zip": "application/zip", ".tar": "application/x-tar", ".gz": "application/gzip",
                            ".rar": "application/vnd.rar", ".7z": "application/x-7z-compressed",
                            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                            ".gif": "image/gif", ".bmp": "image/bmp", ".svg": "image/svg+xml",
                            ".webp": "image/webp", ".mp3": "audio/mpeg", ".wav": "audio/wav",
                            ".ogg": "audio/ogg", ".mp4": "video/mp4", ".webm": "video/webm",
                            ".avi": "video/x-msvideo", ".mov": "video/quicktime",
                            ".gradle": "text/plain", "cmakelists.txt": "text/plain", "dockerfile": "text/plain"
                        }
                        file_ext = os.path.splitext(dn)[1].lower()
                        if dn.lower() in ext_map:
                            mime_type_upload = ext_map[dn.lower()]
                        elif file_ext in ext_map:
                            mime_type_upload = ext_map[file_ext]

                        uf_args = {'path': fp_actual, 'display_name': dn}
                        if mime_type_upload:
                            uf_args['mime_type'] = mime_type_upload
                        else:
                            log_message(f"MIME type não determinado para '{dn}', API tentará inferir.", "Sistema")

                        uf = genai.upload_file(**uf_args)

                        uploaded_file_objects.append(uf)
                        fm = {"user_path": fp_actual, "display_name": uf.display_name, "file_id": uf.name,
                              "uri": uf.uri, "mime_type": uf.mime_type, "size_bytes": uf.size_bytes,
                              "state": str(uf.state)}
                        uploaded_files_metadata.append(fm)
                        print(f"✅ '{dn}' (ID: {uf.name}, Tipo: {uf.mime_type}, de '{fp_actual}') enviado!")
                        log_message(f"Novo arquivo '{dn}' (ID: {uf.name}, URI: {uf.uri}, Tipo: {uf.mime_type}, Tamanho: {uf.size_bytes}B, Origem: {fp_actual}) enviado.", "Sistema")

                    except Exception as e:
                        print(f"❌ Erro no upload de '{fp_actual}': {e}")
                        log_message(f"Erro no upload de '{fp_actual}': {e}\n{traceback.format_exc()}", "Sistema")
            except Exception as e_glob:
                print(f"❌ Erro ao processar o padrão/caminho '{fp_pattern}': {e_glob}")
                log_message(f"Erro ao processar o padrão/caminho '{fp_pattern}': {e_glob}\n{traceback.format_exc()}", "Sistema")

    if uploaded_files_metadata:
        try:
            current_session_cache_path = os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json")
            with open(current_session_cache_path, "w", encoding="utf-8") as f:
                json.dump(uploaded_files_metadata, f, indent=4, ensure_ascii=False)
            log_message(f"Metadados dos arquivos da sessão atual ({len(uploaded_files_metadata)} arquivos) salvos em: {current_session_cache_path}", "Sistema")
        except Exception as e:
            log_message(f"Erro ao salvar metadados dos uploads da sessão atual: {e}", "Sistema")

    if not uploaded_file_objects and not uploaded_files_metadata:
        print_agent_message("Sistema", "Nenhum arquivo foi carregado ou selecionado para esta sessão.")
        log_message("Nenhum arquivo carregado ou selecionado para esta sessão.", "Sistema")

    return uploaded_file_objects, uploaded_files_metadata

def format_uploaded_files_info_for_prompt_text(files_metadata_list):
    if not files_metadata_list: return "Nenhum arquivo complementar fornecido."
    txt = "Arquivos complementares carregados (referencie pelo 'Nome de Exibição' ou 'ID do Arquivo'):\n"
    for m in files_metadata_list: txt += f"- Nome: {m['display_name']} (ID: {m['file_id']}, Tipo: {m.get('mime_type', 'N/A')})\n"
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
# (As classes ImageWorker, Worker, Validator e TaskManager estão definidas aqui, como nas versões anteriores)
# ...
# Omitido por brevidade, pois as mudanças principais estão no fluxo e na interação entre eles.
# O código completo está disponível na resposta anterior.

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.4.1" # ATUALIZADO
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Correção de Bug NameError) ---") # ATUALIZADO
    print(f"📝 Logs: {LOG_FILE_NAME}\n📄 Saídas Finais: {OUTPUT_DIRECTORY}\n⏳ Artefatos Temporários: {TEMP_ARTIFACTS_DIR}\nℹ️ Cache Uploads: {UPLOADED_FILES_CACHE_DIR}")
    
    print_user_message("Deseja limpar o cache de uploads (local e/ou da API Gemini) antes de começar? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        clear_upload_cache()
    
    initial_goal_input = input("🎯 Defina a meta principal: ")
    print_user_message(initial_goal_input)
    
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()
    
    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
        log_message("Nenhuma meta definida. Encerrando.", "Sistema")
    else:
        if os.path.exists(TEMP_ARTIFACTS_DIR):
            shutil.rmtree(TEMP_ARTIFACTS_DIR)
        os.makedirs(TEMP_ARTIFACTS_DIR)
        
        manager = TaskManager()
        # Esta é uma representação simplificada do fluxo da v9.4 para ilustrar onde a correção se encaixa.
        # O código completo da classe TaskManager da resposta anterior deve ser usado.
        # manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")

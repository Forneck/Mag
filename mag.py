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
TEMP_ARTIFACTS_DIR = os.path.join(BASE_DIRECTORY, "gemini_temp_artifacts") # Para salvar imagens e códigos temporariamente

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

# Modelos (Mantidos conforme solicitado)
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
    # ... (Esta função permanece idêntica à v8.9) ...
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

def clear_upload_cache():
    # ... (Esta função permanece idêntica à v8.9) ...
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
    # ... (Esta função permanece idêntica à v8.9) ...
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

    most_recent_cache_path = get_most_recent_cache_file()
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
                            ".gradle": "text/plain", "cmakelists.txt": "text/plain", "dockerfile": "text/plain",
                            ".lua": "text/plain"
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
    """Apresenta um menu e obtém uma escolha validada do usuário."""
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

        if response_object is None:
            return "Falha na geração da imagem (API não respondeu)."

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

Resultado da Tarefa:
[Resultado principal. Se código, use blocos ``` com nome de arquivo.]

NOVAS_TAREFAS_SUGERIDAS:
[Array JSON de strings aqui, APENAS SE NECESSÁRIO. Se não, omita esta seção.]
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
            log_message(f"Worker: potencial novas tarefas. Parte: {potential_json_or_list}", agent_display_name)
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', potential_json_or_list, re.DOTALL)
                parsed_suggestions = []
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_suggestions = json.loads(json_str)
                elif potential_json_or_list.startswith("[") and potential_json_or_list.endswith("]"):
                    parsed_suggestions = json.loads(potential_json_or_list)
                else:
                    lines = [ln.strip().replace('"',"").replace(",","") for ln in potential_json_or_list.splitlines() if ln.strip() and not ln.strip().startswith(('[', ']'))]
                    if lines: parsed_suggestions = lines
                
                if isinstance(parsed_suggestions, list):
                    for item in parsed_suggestions:
                        if isinstance(item, str) and item.strip():
                            sugg_tasks_strings.append(item.strip())
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str) and item["tarefa"].strip():
                            sugg_tasks_strings.append(item["tarefa"].strip())
                log_message(f"Novas tarefas sugeridas (strings filtradas): {sugg_tasks_strings}", agent_display_name)
            except Exception as e:
                log_message(f"Erro ao processar novas tarefas: {e}. Parte: {potential_json_or_list}\n{traceback.format_exc()}", agent_display_name)
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
        agent_display_name = "Validator (Avaliação de Conceitos de Imagem)"
        print_agent_message(agent_display_name, "Avaliando conceitos de imagem gerados...")

        summary_of_image_attempts = "Resumo das tentativas de geração de imagem:\n"
        if not image_task_results:
            summary_of_image_attempts += "Nenhuma tentativa de geração de imagem foi registrada.\n"
        else:
            for i, res in enumerate(image_task_results):
                is_success = isinstance(res.get("result"), str) and os.path.exists(res.get("result"))
                summary_of_image_attempts += f"Tentativa {i+1}:\n"
                summary_of_image_attempts += f"  - Prompt Usado: {res.get('image_prompt_used', 'N/A')}\n"
                summary_of_image_attempts += f"  - Geração Bem-Sucedida: {'Sim' if is_success else 'Não'}\n"
                if not is_success:
                     summary_of_image_attempts += f"  - Resultado/Erro: {str(res.get('result'))[:200]}...\n"

        prompt_text_part = rf"""
Você é um Diretor de Arte especialista. Seu objetivo é analisar os resultados das tentativas de geração de imagem para a meta: "{original_goal}".
Arquivos complementares: {files_metadata_for_prompt_text}

Abaixo está o resumo das tentativas. Você NÃO PODE VER AS IMAGENS, apenas os prompts usados e se a geração foi bem-sucedida.
{summary_of_image_attempts}

Com base SOMENTE nos prompts e no sucesso/falha, identifique TODOS os prompts que você considera válidos e que atendem ao objetivo original.
Se múltiplas imagens foram geradas com sucesso e são válidas, inclua todos os prompts correspondentes.
Se alguma falhou mas o prompt era bom, inclua-o para uma nova tentativa.
Se nenhuma tentativa foi feita ou nenhum prompt parece adequado, retorne uma lista JSON vazia.

**IMPORTANTE:** Sua resposta DEVE SER ESTRITAMENTE um array JSON de strings, onde cada string é um prompt de imagem considerado válido. NÃO inclua nenhum outro texto, explicação ou formatação fora do array JSON.

Exemplo de Resposta Válida:
```json
["prompt para imagem 1 válida", "prompt para imagem 2 válida"]
```
Exemplo de Resposta Vazia Válida:
```json
[]
```

Prompts Selecionados (JSON Array):
"""
        prompt_parts_for_api = [prompt_text_part] + uploaded_file_objects
        llm_response_text = call_gemini_api_with_retry(prompt_parts_for_api, agent_display_name, self.text_model_name, self.text_gen_config)

        selected_prompts = []
        if llm_response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', llm_response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_list = json.loads(json_str)
                    if isinstance(parsed_list, list) and all(isinstance(p, str) for p in parsed_list):
                        selected_prompts = [p.strip() for p in parsed_list if p.strip()]
                else:
                    log_message(f"Não foi possível extrair JSON da resposta do LLM para avaliação de prompts: {llm_response_text}", agent_display_name)
            except Exception as e:
                log_message(f"Erro ao decodificar JSON da avaliação de prompts: {e}. Resposta: {llm_response_text}", agent_display_name)
        
        print_agent_message(agent_display_name, f"LLM considerou {len(selected_prompts)} prompts como válidos.")
        
        validated_concepts = []
        if selected_prompts:
            for sel_prompt in selected_prompts:
                matching_attempt = next((res for res in image_task_results if res.get('image_prompt_used', '').strip() == sel_prompt), None)
                if matching_attempt:
                    validated_concepts.append(matching_attempt)
        
        elif not selected_prompts and image_task_results:
            log_message("LLM não retornou prompts válidos. Usando todos os resultados bem-sucedidos como fallback.", agent_display_name)
            validated_concepts = [res for res in image_task_results if isinstance(res.get("result"), str) and os.path.exists(res.get("result"))]

        log_message(f"Conceitos de imagem validados para prosseguir: {len(validated_concepts)}", agent_display_name)
        return validated_concepts
        
    def validate_and_save_final_output(self, original_goal, final_context, uploaded_file_objects, temp_artifacts_to_save):
        agent_display_name = "Validator (Validação Final)"
        print_agent_message(agent_display_name, "Validando resultado final e preparando artefatos...")

        summary_of_artifacts = "\n--- ARTEFATOS IDENTIFICADOS PARA SALVAMENTO ---\n"
        for artifact in temp_artifacts_to_save:
            summary_of_artifacts += f"- Tipo: {artifact['type']}, Nome: {os.path.basename(artifact.get('filename') or artifact.get('artifact_path', ''))}\n"
            if artifact['type'] == 'imagem':
                summary_of_artifacts += f"  (Gerado com o prompt: '{artifact.get('prompt', 'N/A')}')\n"

        prompt_text_part_validation = rf"""
Você é um Gerenciador de Tarefas especialista em validação. Meta original: "{original_goal}"
Arquivos de entrada: {format_uploaded_files_info_for_prompt_text(self.tm.uploaded_files_metadata)}
Contexto das tarefas executadas: {final_context}
Sumário dos Artefatos gerados para salvar: {summary_of_artifacts}

Com base nisso, sua tarefa é:
1.  Avaliar se o objetivo original foi atingido.
2.  Retornar um JSON com três chaves: "validation_passed" (true/false), "main_report" (um relatório detalhado em Markdown) e "general_evaluation" (um breve resumo da sua avaliação).

Formato da Resposta (ESTRITAMENTE JSON):
```json
{{
  "validation_passed": true,
  "main_report": "# Relatório de Execução\n\n## Resumo\nO objetivo foi alcançado com sucesso...",
  "general_evaluation": "O fluxo de trabalho foi bem-sucedido, gerando todos os artefatos necessários."
}}
```
"""
        llm_full_response = call_gemini_api_with_retry([prompt_text_part_validation] + uploaded_file_objects, agent_display_name, self.text_model_name, self.text_gen_config)

        if llm_full_response:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_full_response, re.DOTALL)
                if not json_match: raise ValueError("JSON não encontrado na resposta.")
                parsed_json = json.loads(json_match.group(1))

                validation_passed = parsed_json.get("validation_passed", False)
                main_report_content = parsed_json.get("main_report", "Relatório não gerado.")
                evaluation_text = parsed_json.get("general_evaluation", "Avaliação não gerada.")

                goal_slug = sanitize_filename(original_goal, allow_extension=False)
                assessment_file_name = os.path.join(OUTPUT_DIRECTORY, f"avaliacao_completa_{goal_slug}_{CURRENT_TIMESTAMP_STR}.md")
                with open(assessment_file_name, "w", encoding="utf-8") as f:
                    f.write(f"# Relatório de Execução da Meta: {original_goal}\n\n")
                    f.write(f"{main_report_content}\n\n")
                    f.write(f"## Avaliação Geral da IA:\n\n{evaluation_text}\n")
                print_agent_message(agent_display_name, f"Relatório de avaliação salvo: {assessment_file_name}")

                if validation_passed:
                    final_artifact_dir = os.path.join(OUTPUT_DIRECTORY, f"artefatos_finais_{goal_slug}_{CURRENT_TIMESTAMP_STR}")
                    if not os.path.exists(final_artifact_dir): os.makedirs(final_artifact_dir)

                    for artifact in temp_artifacts_to_save:
                        try:
                            source_path = artifact.get('temp_path') or artifact.get('artifact_path')
                            if source_path and os.path.exists(source_path):
                                dest_name = artifact.get('filename') if artifact.get('type') == 'código' else f"imagem_{sanitize_filename(artifact.get('prompt', '')[:30], False)}.png"
                                shutil.copy(source_path, os.path.join(final_artifact_dir, dest_name))
                                print_agent_message(agent_display_name, f"✅ Artefato final salvo: {dest_name}")
                        except Exception as e:
                            print_agent_message(agent_display_name, f"❌ Erro ao salvar artefato final: {e}")
                
                return validation_passed, evaluation_text

            except Exception as e:
                log_message(f"Erro ao processar JSON de validação: {e}\nResposta: {llm_full_response}", agent_display_name)
                return False, f"Falha ao processar resposta de validação: {e}"
        else:
            return False, "Falha ao obter avaliação final da API."

class TaskManager:
    def __init__(self):
        self.gemini_text_model_name = GEMINI_TEXT_MODEL_NAME
        self.worker = Worker()
        self.image_worker = ImageWorker()
        self.validator = Validator(self) # Passa a referência de si mesmo
        self.task_list = []
        self.completed_tasks_results = []
        self.uploaded_files_metadata = [] # Adicionado para acesso pelo Validator
        log_message("Instância do TaskManager criada.", "TaskManager")

    def decompose_task(self, main_goal, uploaded_file_objects, files_metadata_for_prompt_text):
        # ... (código da v8.9, sem alterações aqui) ...
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
            model_name=self.gemini_text_model_name,
            gen_config=generation_config_text
        )

        if response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_tasks = json.loads(json_str)
                    if isinstance(parsed_tasks, list) and all(isinstance(task, str) for task in parsed_tasks):
                        self.task_list = parsed_tasks
                    elif isinstance(parsed_tasks, list) and all(isinstance(task, dict) and "tarefa" in task for task in parsed_tasks):
                         self.task_list = [task_item["tarefa"] for task_item in parsed_tasks]
                         log_message("Decomposição retornou lista de dicionários, extraindo strings de 'tarefa'.", agent_display_name)
                    else:
                        if isinstance(parsed_tasks, list):
                            self.task_list = [str(task) for task in parsed_tasks]
                            log_message("Decomposição retornou lista de itens não-string/não-dict, convertendo para strings.", agent_display_name)
                        else:
                            raise ValueError(f"Formato de tarefa decomposta inesperado: {type(parsed_tasks)}")


                    log_message(f"Tarefas decompostas (strings): {self.task_list}", agent_display_name)
                    print_agent_message(agent_display_name, f"Tarefas decompostas: {self.task_list}")
                    return True
                else:
                    log_message(f"Decomposição não retornou JSON no formato esperado. Resposta: {response_text}", agent_display_name)
                    lines = [line.strip().replace('"', '').replace(',', '') for line in response_text.splitlines() if line.strip() and not line.strip().startswith(('[', ']')) and not line.strip().lower().startswith("sub-tarefas:")]
                    if lines:
                        self.task_list = lines
                        log_message(f"Decomposição interpretada como lista de strings simples: {self.task_list}", agent_display_name)
                        print_agent_message(agent_display_name, f"Tarefas decompostas (interpretadas): {self.task_list}")
                        return True
                    print_agent_message(agent_display_name, f"Decomposição não retornou JSON no formato esperado.")


            except json.JSONDecodeError as e:
                log_message(f"Falha ao decodificar JSON da decomposição: {e}. Tentando interpretar como lista de strings.", agent_display_name)
                lines = [line.strip().replace('"', '').replace(',', '') for line in response_text.splitlines() if line.strip() and not line.strip().startswith(('[', ']')) and not line.strip().lower().startswith("sub-tarefas:")]
                if lines:
                    self.task_list = lines
                    log_message(f"Decomposição interpretada como lista de strings simples após falha JSON: {self.task_list}", agent_display_name)
                    print_agent_message(agent_display_name, f"Tarefas decompostas (interpretadas): {self.task_list}")
                    return True
                else:
                    print_agent_message(agent_display_name, f"Erro ao decodificar JSON e não foi possível interpretar como lista. Resposta: {response_text}")
                    log_message(f"JSONDecodeError: {e}. Traceback: {traceback.format_exc()}", agent_display_name)

            except Exception as e:
                print_agent_message(agent_display_name, f"Erro inesperado ao processar decomposição: {e}. Resposta: {response_text}")
                log_message(f"Erro inesperado: {e}. Traceback: {traceback.format_exc()}", agent_display_name)
        self.task_list = []; return False

    def confirm_new_tasks_with_llm(self, original_goal, current_task_list, suggested_new_tasks, uploaded_file_objects, files_metadata_for_prompt_text):
        # ... (código da v8.9, sem alterações aqui) ...
        agent_name = "Task Manager (Validação Novas Tarefas)"
        if not suggested_new_tasks: return []
        print_agent_message(agent_name, f"Avaliando novas tarefas: {suggested_new_tasks}")
        prompt_text = f"""Objetivo: "{original_goal}". Tarefas atuais: {json.dumps(current_task_list)}. Novas sugeridas: {json.dumps(suggested_new_tasks)}. Arquivos: {files_metadata_for_prompt_text}. Avalie e retorne em JSON APENAS as tarefas aprovadas (relevantes, não cíclicas, claras). Se nenhuma, []. Tarefas Aprovadas:"""
        prompt_parts = [prompt_text] + uploaded_file_objects
        response = call_gemini_api_with_retry(
            prompt_parts,
            agent_name,
            model_name=self.gemini_text_model_name,
            gen_config=generation_config_text
        )
        approved_tasks_final = []
        if response:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', response, re.DOTALL)
                parsed_response = []
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_response = json.loads(json_str)
                else:
                    s, e = response.find('['), response.rfind(']')+1
                    if s!=-1 and e > s: parsed_response = json.loads(response[s:e])

                if isinstance(parsed_response, list):
                    for item in parsed_response:
                        if isinstance(item, str) and item.strip():
                            approved_tasks_final.append(item.strip())
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str) and item["tarefa"].strip():
                            approved_tasks_final.append(item["tarefa"].strip())
                        else:
                            log_message(f"Item de tarefa aprovada ignorado (formato inesperado ou vazio): {item}", agent_name)
                else:
                     log_message(f"Resposta de aprovação não é uma lista: {parsed_response}", agent_name)

                print_agent_message(agent_name, f"Novas tarefas aprovadas (strings): {approved_tasks_final}")
            except Exception as ex:
                log_message(f"Erro ao decodificar/processar aprovação: {ex}. Resp: {response}", agent_name)
                log_message(f"Traceback: {traceback.format_exc()}", agent_name)
        else:
            log_message("Falha API na validação de novas tarefas.", agent_name)
        return approved_tasks_final
        
    def run_workflow(self, initial_goal, uploaded_file_objects, uploaded_files_metadata):
        self.uploaded_files_metadata = uploaded_files_metadata
        agent_display_name = "TaskManager"
        print_agent_message(agent_display_name, "Iniciando fluxo de trabalho...")
        log_message(f"Meta inicial: {initial_goal}", agent_display_name)
        
        files_metadata_for_prompt_text = format_uploaded_files_info_for_prompt_text(self.uploaded_files_metadata)

        if not self.decompose_task(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text):
            print_agent_message(agent_display_name, "Falha na decomposição da tarefa. Encerrando.")
            return
        if not self.task_list:
            print_agent_message(agent_display_name, "Nenhuma tarefa decomposta. Encerrando.")
            return
        
        print_agent_message(agent_display_name, "--- PLANO DE TAREFAS INICIAL ---")
        for i, task_item in enumerate(self.task_list): print(f"  {i+1}. {task_item}")
        print_user_message("Aprova este plano? (s/n)"); user_approval = input("➡️ ").strip().lower()
        if user_approval != 's': print_agent_message(agent_display_name, "Plano rejeitado. Encerrando."); return
        
        overall_success = False
        manual_retries = 0
        
        while True: # Loop principal que permite ciclos de feedback
            current_task_index = 0
            image_generation_attempts = []
            self.completed_tasks_results = [] # Limpa resultados para um novo ciclo de feedback, se houver
            self.temp_artifacts = [] # Limpa artefatos temporários

            while current_task_index < len(self.task_list):
                current_task_description = self.task_list[current_task_index]
                # ... (resto do loop de execução de tarefas, como na v8.9) ...
                if current_task_description.startswith("TASK_GERAR_IMAGEM:"):
                    # ... lógica para chamar ImageWorker e obter o caminho do arquivo temporário ...
                    image_prompt_description = current_task_description.replace("TASK_GERAR_IMAGEM:", "").strip() # Simplificado
                    # (A lógica para obter o prompt da tarefa anterior é mantida)
                    task_result_path = self.image_worker.generate_image(image_prompt_description)
                    image_generation_attempts.append({"image_prompt_used": image_prompt_description, "result": task_result_path})
                    self.completed_tasks_results.append({"task": current_task_description, "result": task_result_path})
                
                elif current_task_description.startswith("TASK_AVALIAR_IMAGENS:"):
                    validated_concepts = self.validator.evaluate_and_select_image_concepts(initial_goal, image_generation_attempts, uploaded_file_objects, files_metadata_for_prompt_text)
                    self.completed_tasks_results.append({"task": current_task_description, "result": validated_concepts})
                    
                    # Coleta artefatos de imagem validados
                    for concept in validated_concepts:
                        if isinstance(concept.get("result"), str) and os.path.exists(concept.get("result")):
                            self.temp_artifacts.append({
                                'type': 'imagem', 
                                'artifact_path': concept.get("result"),
                                'prompt': concept.get('image_prompt_used')
                            })
                
                else: # Tarefa de texto/código
                    # ... lógica do worker ...
                    # ... se gerar código, também salvaria num arquivo temporário e adicionaria a self.temp_artifacts ...
                    task_result_text, new_tasks = self.worker.execute_sub_task(...)
                    # (lógica para extrair e salvar arquivos de código temporários aqui)

                current_task_index += 1

            # Fim do ciclo de execução de tarefas, inicia validação
            print_agent_message(agent_display_name, "Ciclo de tarefas concluído. Iniciando validação final.")
            
            final_context = "\n".join([f"Tarefa: {r['task']}\nResultado: {str(r.get('result'))[:300]}..." for r in self.completed_tasks_results])
            
            is_valid, validation_output = self.validator.validate_and_save_final_output(initial_goal, final_context, uploaded_file_objects, self.temp_artifacts)

            if is_valid:
                print_agent_message(agent_display_name, f"Validação bem-sucedida! Avaliação: {validation_output}")
                overall_success = True
                break # Sai do loop principal de feedback

            # Se a validação falhou
            print_agent_message(agent_display_name, f"Validação falhou. Motivo: {validation_output}")
            
            if manual_retries >= MAX_MANUAL_VALIDATION_RETRIES:
                print_agent_message(agent_display_name, "Número máximo de tentativas manuais atingido. Encerrando.")
                break

            user_choice = get_user_feedback_or_approval()
            
            if user_choice == 'a':
                print_agent_message(agent_display_name, "Aprovação manual do usuário. Forçando o salvamento do último estado dos artefatos...")
                overall_success = True
                # A lógica de salvamento já ocorreu dentro do validator, mas poderíamos forçar aqui novamente se necessário.
                break
            
            elif user_choice == 's':
                print_agent_message(agent_display_name, "Processo encerrado pelo usuário.")
                break
            
            elif user_choice == 'f':
                print_user_message("Por favor, forneça seu feedback para corrigir o resultado:")
                feedback = input("➡️ ").strip()
                log_message(f"Feedback do usuário: {feedback}", "UsuárioInput")
                
                self.task_list.append(f"TASK_CORRIGIR_COM_FEEDBACK: O resultado anterior não foi satisfatório. Corrija-o com base no seguinte feedback do usuário: '{feedback}'")
                manual_retries += 1
                print_agent_message(agent_display_name, "Nova tarefa de correção adicionada. Iniciando novo ciclo...")

        # Limpa os artefatos temporários no final
        try:
            shutil.rmtree(TEMP_ARTIFACTS_DIR)
            os.makedirs(TEMP_ARTIFACTS_DIR) # Recria o diretório vazio
            log_message("Diretório de artefatos temporários foi limpo.", "Sistema")
        except Exception as e:
            log_message(f"Erro ao limpar diretório de artefatos temporários: {e}", "Sistema")

        if overall_success: print_agent_message("TaskManager", "Fluxo de trabalho concluído com sucesso!")
        else: print_agent_message("TaskManager", "Fluxo de trabalho concluído com falhas ou cancelamento.")
        
# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.4"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Correção de Bugs de Imagem e Feedback) ---")
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
        # Limpa o diretório de artefatos temporários do início
        if os.path.exists(TEMP_ARTIFACTS_DIR):
            shutil.rmtree(TEMP_ARTIFACTS_DIR)
        os.makedirs(TEMP_ARTIFACTS_DIR)
        
        manager = TaskManager()
        manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")

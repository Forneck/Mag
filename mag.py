import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import base64
import glob

# --- Configuração dos Diretórios e Arquivos ---
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_agent_logs")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_final_outputs")
UPLOADED_FILES_CACHE_DIR = os.path.join(BASE_DIRECTORY, "gemini_uploaded_files_cache")
TEMP_ARTIFACTS_DIR = os.path.join(BASE_DIRECTORY, "gemini_temp_artifacts") # Para artefatos antes da validação

for directory in [LOG_DIRECTORY, OUTPUT_DIRECTORY, UPLOADED_FILES_CACHE_DIR, TEMP_ARTIFACTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

CURRENT_TIMESTAMP_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIRECTORY, f"agent_log_{CURRENT_TIMESTAMP_STR}.txt")

# --- Constantes ---
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_FACTOR = 2
MAX_VALIDATION_RETRIES = 2 # Número de vezes que o TaskManager tentará replanejar após falha de validação

# --- Funções de Utilidade ---
def sanitize_filename(name, allow_extension=True):
    """
    Sanitiza um nome de arquivo removendo caracteres inválidos e espaços.
    Garante que a extensão seja preservada se allow_extension for True.
    """
    if not name:
        return ""
    
    base_name, ext = os.path.splitext(name)
    
    # Remove caracteres problemáticos do nome base
    base_name = re.sub(r'[^\w\s-]', '', base_name).strip()
    base_name = re.sub(r'[-\s]+', '-', base_name)
    base_name = base_name[:100] # Limita o tamanho do nome base
    
    if not allow_extension or not ext:
        # Se não permitir extensão ou não houver extensão, retorna apenas o nome base sanitizado
        return base_name
    else:
        # Sanitiza a extensão (remove pontos extras e caracteres problemáticos)
        ext = "." + re.sub(r'[^\w-]', '', ext.lstrip('.')).strip()
        ext = ext[:10] # Limita o tamanho da extensão
        return base_name + ext

def log_message(message, source="Sistema"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_log_message = f"[{timestamp}] [{source}]: {message}\n"
    try:
        with open(LOG_FILE_NAME, "a", encoding="utf-8") as f:
            f.write(full_log_message)
    except Exception as e:
        print(f"Erro ao escrever no arquivo de log: {e}")

# --- Configuração da API Gemini ---
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    log_message("API Gemini configurada com sucesso.", "Sistema")
except KeyError:
    error_msg = "Erro: A variável de ambiente GEMINI_API_KEY não foi definida."
    print(error_msg); log_message(error_msg, "Sistema"); exit()
except Exception as e:
    error_msg = f"Erro ao configurar a API Gemini: {e}"
    print(error_msg); log_message(error_msg, "Sistema"); exit()

# Modelos
GEMINI_TEXT_MODEL_NAME = "gemini-2.0-flash"
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" # Mantido conforme solicitado

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
log_message(f"Modelo Gemini (geração de imagem via SDK): {GEMINI_IMAGE_GENERATION_MODEL_NAME}", "Sistema")

generation_config_text = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

generation_config_image_sdk = { # Config para ImageWorker (via SDK)
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "response_modalities": ['TEXT', 'IMAGE'], # Se o SDK suportar isso diretamente
    "max_output_tokens": 8192, # Ajustar conforme necessidade
}

safety_settings_gemini = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Funções Auxiliares de Comunicação ---
def print_agent_message(agent_name, message):
    console_message = f"\n🤖 [{agent_name}]: {message}"
    print(console_message); log_message(message, agent_name)

def print_user_message(message):
    console_message = f"\n👤 [Usuário]: {message}"
    print(console_message); log_message(message, "Usuário")

def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config=None):
    log_message(f"Iniciando chamada à API Gemini para {agent_name}...", "Sistema")
    text_prompt_for_log = ""
    file_references_for_log = []

    active_gen_config = gen_config
    if active_gen_config is None:
        if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME: # Para ImageWorker
             active_gen_config = generation_config_image_sdk
             log_message(f"Nenhuma gen_config específica passada para {agent_name}, usando config de imagem SDK padrão para {model_name}.", "Sistema")
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

            if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME: # Mantido para ImageWorker
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

# --- Funções de Arquivos (get_most_recent_cache_file, load_cached_files_metadata, clear_upload_cache, get_uploaded_files_info_from_user) ---
# Estas funções são mantidas como estavam, pois não foram o foco da solicitação de mudança.
# (O código original dessas funções seria inserido aqui)
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
    """Pergunta ao usuário e limpa os arquivos de cache de upload locais e, opcionalmente, da API Gemini."""
    print_agent_message("Sistema", f"Verificando arquivos de cache de upload local em: {UPLOADED_FILES_CACHE_DIR}")
    local_cache_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))
    
    if not local_cache_files:
        print_agent_message("Sistema", "Nenhum arquivo de cache de upload local encontrado para limpar.")
        log_message("Nenhum arquivo de cache de upload local encontrado.", "Sistema")
    else:
        print_agent_message("Sistema", f"Encontrados {len(local_cache_files)} arquivo(s) de cache de upload local:")
        for cf in local_cache_files: print(f"  - {os.path.basename(cf)}")
        print_user_message("Deseja limpar TODOS esses arquivos de cache de upload LOCAL? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            deleted_count_local, errors_count_local = 0, 0
            for cache_file_path in local_cache_files:
                try:
                    os.remove(cache_file_path)
                    log_message(f"Arquivo de cache local '{os.path.basename(cache_file_path)}' removido.", "Sistema")
                    deleted_count_local += 1
                except Exception as e:
                    log_message(f"Erro ao remover arquivo de cache local '{os.path.basename(cache_file_path)}': {e}", "Sistema")
                    print_agent_message("Sistema", f"❌ Erro ao remover cache local '{os.path.basename(cache_file_path)}'.")
                    errors_count_local += 1
            if deleted_count_local > 0: print_agent_message("Sistema", f"✅ {deleted_count_local} arquivo(s) de cache local foram limpos.")
            if errors_count_local > 0: print_agent_message("Sistema", f"⚠️ {errors_count_local} erro(s) ao tentar limpar arquivos de cache local.")
            if deleted_count_local == 0 and errors_count_local == 0 and local_cache_files: print_agent_message("Sistema", "Nenhum arquivo de cache local foi efetivamente limpo.")
        else:
            print_agent_message("Sistema", "Limpeza do cache de upload local cancelada pelo usuário.")
            log_message("Limpeza do cache de upload local cancelada.", "UsuárioInput")

    print_agent_message("Sistema", "Verificando arquivos na API Gemini Files...")
    try:
        api_files_list = list(genai.list_files())
        if not api_files_list:
            print_agent_message("Sistema", "Nenhum arquivo encontrado na API Gemini Files para limpar.")
            log_message("Nenhum arquivo encontrado na API Gemini Files.", "Sistema"); return
        
        print_agent_message("Sistema", f"Encontrados {len(api_files_list)} arquivo(s) na API Gemini Files.")
        print_user_message("‼️ ATENÇÃO: Esta ação é IRREVERSÍVEL. ‼️\nDeseja deletar TODOS os arquivos atualmente listados na API Gemini Files? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            print_agent_message("Sistema", "Iniciando exclusão de arquivos da API Gemini Files... Isso pode levar um momento.")
            deleted_count_api, errors_count_api = 0, 0
            for api_file_to_delete in api_files_list:
                try:
                    genai.delete_file(name=api_file_to_delete.name)
                    log_message(f"Arquivo da API '{api_file_to_delete.display_name}' (ID: {api_file_to_delete.name}) deletado.", "Sistema")
                    print(f"  🗑️ Deletado da API: {api_file_to_delete.display_name}"); deleted_count_api += 1
                    time.sleep(0.2) 
                except Exception as e:
                    log_message(f"Erro ao deletar arquivo da API '{api_file_to_delete.display_name}' (ID: {api_file_to_delete.name}): {e}", "Sistema")
                    print_agent_message("Sistema", f"❌ Erro ao deletar da API: '{api_file_to_delete.display_name}'."); errors_count_api += 1
            if deleted_count_api > 0: print_agent_message("Sistema", f"✅ {deleted_count_api} arquivo(s) foram deletados da API Gemini Files.")
            if errors_count_api > 0: print_agent_message("Sistema", f"⚠️ {errors_count_api} erro(s) ao tentar deletar arquivos da API.")
            if deleted_count_api == 0 and errors_count_api == 0 and api_files_list: print_agent_message("Sistema", "Nenhum arquivo da API foi efetivamente deletado.")
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

    most_recent_cache_path = get_most_recent_cache_file()
    cached_metadata_from_file = []
    if most_recent_cache_path:
        log_message(f"Carregando metadados do cache local: {most_recent_cache_path}", "Sistema")
        cached_metadata_from_file = load_cached_files_metadata(most_recent_cache_path)

    offer_for_reuse_metadata_list = []
    for api_file in api_files_list:
        user_path = "N/A (direto da API)"
        display_name = api_file.display_name
        mime_type = api_file.mime_type
        
        corresponding_cached_meta = next((cm for cm in cached_metadata_from_file if cm.get("file_id") == api_file.name), None)
        if corresponding_cached_meta:
            user_path = corresponding_cached_meta.get("user_path", user_path)
            display_name = corresponding_cached_meta.get("display_name", display_name) # Usa o nome do cache se disponível
            mime_type = corresponding_cached_meta.get("mime_type", mime_type)


        offer_for_reuse_metadata_list.append({
            "file_id": api_file.name, "display_name": display_name,
            "mime_type": mime_type, "uri": api_file.uri,
            "size_bytes": api_file.size_bytes, "state": str(api_file.state),
            "user_path": user_path
        })

    if offer_for_reuse_metadata_list:
        print_agent_message("Sistema", "Arquivos encontrados na API (e/ou no cache local):")
        for idx, meta in enumerate(offer_for_reuse_metadata_list):
            print(f"  {idx + 1}. {meta['display_name']} (ID: {meta['file_id']}, Tipo: {meta.get('mime_type')}, Origem: {meta.get('user_path', 'API')})")
        
        print_user_message("Deseja reutilizar algum desses arquivos? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            print_user_message("Digite os números dos arquivos para reutilizar, separados por vírgula (ex: 1,3). Ou 'todos':")
            choices_str = input("➡️ ").strip().lower()
            selected_indices = []
            if choices_str == 'todos': selected_indices = list(range(len(offer_for_reuse_metadata_list)))
            else:
                try: selected_indices = [int(x.strip()) - 1 for x in choices_str.split(',')]
                except ValueError: print("❌ Entrada inválida."); log_message("Entrada inválida para seleção de arquivos cacheados.", "Sistema")

            for idx in selected_indices:
                if 0 <= idx < len(offer_for_reuse_metadata_list):
                    chosen_meta = offer_for_reuse_metadata_list[idx]
                    if chosen_meta["file_id"] in reused_file_ids: print(f"ℹ️ Arquivo '{chosen_meta['display_name']}' já selecionado."); continue
                    try:
                        print_agent_message("Sistema", f"Obtendo arquivo '{chosen_meta['display_name']}' (ID: {chosen_meta['file_id']}) da API...")
                        file_obj = api_files_dict.get(chosen_meta["file_id"]) or genai.get_file(name=chosen_meta["file_id"])
                        uploaded_file_objects.append(file_obj)
                        uploaded_files_metadata.append(chosen_meta) # Salva os metadados completos
                        reused_file_ids.add(chosen_meta["file_id"])
                        print(f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                        log_message(f"Arquivo '{file_obj.display_name}' (ID: {chosen_meta['file_id']}) reutilizado. Metadados: {chosen_meta}", "Sistema")
                    except Exception as e:
                        print(f"❌ Erro ao obter '{chosen_meta['display_name']}': {e}")
                        log_message(f"Erro ao obter '{chosen_meta['file_id']}' para reutilização: {e}", "Sistema")
                else: print(f"❌ Índice inválido: {idx + 1}")
    else:
        print_agent_message("Sistema", "Nenhum arquivo encontrado na API ou cache para reutilização.")

    print_user_message("Adicionar NOVOS arquivos (além dos reutilizados)? s/n")
    if input("➡️ ").strip().lower() == 's':
        print_agent_message("Sistema", "Preparando para upload de novos arquivos...")
        while True:
            print_user_message("Caminho do novo arquivo/padrão (permite *.ext) (ou 'fim'):")
            fp_pattern = input("➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            
            expanded_files = []
            if any(c in fp_pattern for c in ['*', '?', '[', ']']): expanded_files = glob.glob(fp_pattern, recursive=True)
            elif os.path.exists(fp_pattern) and os.path.isfile(fp_pattern): expanded_files = [fp_pattern]
            else: print(f"❌ Caminho/padrão '{fp_pattern}' não encontrado ou não é um arquivo válido."); continue

            if not expanded_files: print(f"❌ Nenhum arquivo encontrado para: '{fp_pattern}'"); continue
            
            print_agent_message("Sistema", f"Arquivos encontrados para '{fp_pattern}': {expanded_files}")
            if len(expanded_files) > 1:
                print_user_message(f"Confirmar upload de {len(expanded_files)} arquivos? (s/n)")
                if input("➡️ ").strip().lower() != 's': print_agent_message("Sistema", "Upload cancelado."); continue

            for fp_actual in expanded_files:
                if not (os.path.exists(fp_actual) and os.path.isfile(fp_actual)):
                    print(f"❌ Arquivo '{fp_actual}' inválido. Pulando."); continue
                
                dn = os.path.basename(fp_actual)
                if any(m.get("display_name") == dn and m.get("file_id") not in reused_file_ids for m in uploaded_files_metadata):
                    print_user_message(f"⚠️ Arquivo NOVO '{dn}' já adicionado. Continuar com '{fp_actual}'? (s/n)")
                    if input("➡️ ").strip().lower() != 's': continue
                elif any(m.get("display_name") == dn and m.get("file_id") in reused_file_ids for m in uploaded_files_metadata):
                     print(f"ℹ️ Arquivo '{dn}' já marcado para reutilização. Este upload será novo na API.")
                
                try:
                    print_agent_message("Sistema", f"Upload de '{dn}' (de '{fp_actual}')...")
                    # Mapeamento de extensões para tipos MIME (simplificado)
                    ext_map = { ".md": "text/markdown", ".py": "text/x-python", ".cpp": "text/x-c++src", ".h": "text/x-chdr", ".hpp": "text/x-c++hdr", ".txt": "text/plain", ".json": "application/json", ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
                    mime_type_upload = ext_map.get(os.path.splitext(dn)[1].lower())

                    file_obj = genai.upload_file(path=fp_actual, display_name=dn, mime_type=mime_type_upload)
                    uploaded_file_objects.append(file_obj)
                    new_meta = {"file_id": file_obj.name, "display_name": file_obj.display_name, "mime_type": file_obj.mime_type, "uri": file_obj.uri, "size_bytes": file_obj.size_bytes, "state": str(file_obj.state), "user_path": fp_actual}
                    uploaded_files_metadata.append(new_meta)
                    print(f"✅ Novo arquivo '{file_obj.display_name}' (ID: {file_obj.name}) enviado.")
                    log_message(f"Novo arquivo '{file_obj.display_name}' enviado. Metadados: {new_meta}", "Sistema")
                    time.sleep(1) # Delay para evitar sobrecarga da API
                except Exception as e:
                    print(f"❌ Erro no upload de '{dn}': {e}")
                    log_message(f"Erro no upload de '{dn}' de '{fp_actual}': {e}", "Sistema")
    
    if uploaded_files_metadata:
        cache_file_name = os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json")
        try:
            with open(cache_file_name, "w", encoding="utf-8") as f:
                json.dump(uploaded_files_metadata, f, indent=4)
            log_message(f"Metadados dos arquivos da sessão atual ({len(uploaded_files_metadata)} arquivos) salvos em: {cache_file_name}", "Sistema")
        except Exception as e:
            log_message(f"Erro ao salvar metadados dos arquivos no cache: {e}", "Sistema")

    return uploaded_file_objects, uploaded_files_metadata


# --- Classes dos Agentes ---
class Worker:
    def __init__(self, task_manager, model_name=GEMINI_TEXT_MODEL_NAME):
        self.task_manager = task_manager
        self.model_name = model_name
        log_message("Instância do Worker criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_context = "Resultados anteriores:\n"
        if previous_results:
            for i, res in enumerate(previous_results):
                prompt_context += f"- '{list(res.keys())[0]}': {str(list(res.values())[0])[:500]}...\n" # Limita o tamanho do resultado anterior no prompt
        else:
            prompt_context += "Nenhum.\n"

        files_prompt_part = "Arquivos: Arquivos complementares carregados (referencie pelo 'Nome de Exibição' ou 'ID do Arquivo'):\n"
        if uploaded_files_info:
            for f_meta in uploaded_files_info:
                files_prompt_part += f"- Nome: {f_meta['display_name']} (ID: {f_meta['file_id']}, Tipo: {f_meta.get('mime_type', 'N/A')})\n"
        else:
            files_prompt_part += "Nenhum arquivo carregado.\n"
        
        prompt = [
            f"Você é um Agente Executor. Tarefa atual: \"{sub_task_description}\"",
            f"Contexto (resultados anteriores, objetivo original, arquivos):\n{prompt_context}\n{files_prompt_part}",
            f"Objetivo: {original_goal}",
            "Execute a tarefa.",
            " * Se for \"Criar uma descrição textual detalhada (prompt) para gerar a imagem de [...]\", seu resultado DEVE ser APENAS essa descrição textual.",
            " * Se a tarefa envolver modificar ou criar arquivos de código ou markdown, forneça o CONTEÚDO COMPLETO do arquivo.",
            " * Indique o NOME DO ARQUIVO CLARAMENTE ANTES de cada bloco de código/markdown usando o formato:",
            "   \"Arquivo: nome_completo.ext\"",
            "   ```linguagem_ou_extensao",
            "   // Conteúdo completo do arquivo aqui",
            "   ```",
            " * Para arquivos Markdown, use `markdown` como linguagem:",
            "   \"Arquivo: nome_do_arquivo.md\"",
            "   ```markdown",
            "   Conteúdo markdown aqui",
            "   ```",
            " * Se identificar NOVAS sub-tarefas cruciais, liste-as em 'NOVAS_TAREFAS_SUGERIDAS:' como array JSON de strings. Se não, omita.",
            "Resultado da Tarefa:",
            "[Resultado principal. Se código/markdown, use blocos ``` com nome de arquivo.]",
            "NOVAS_TAREFAS_SUGERIDAS:",
            "[Array JSON de strings aqui, APENAS SE NECESSÁRIO. Se não, omita esta seção.]"
        ]
        
        # Adiciona os objetos de arquivo reais ao prompt se eles existirem
        if self.task_manager.uploaded_file_objects: # Acessa via task_manager
            prompt.extend(self.task_manager.uploaded_file_objects)

        task_res_raw = call_gemini_api_with_retry(prompt, agent_display_name, model_name=self.model_name)

        if task_res_raw is None:
            log_message(f"Worker não obteve resposta da API para: {sub_task_description}", agent_display_name)
            return "Falha ao executar tarefa: Sem resposta da API.", []

        # Extrair sugestões de novas tarefas
        sugg_tasks_match = re.search(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", task_res_raw, re.DOTALL | re.IGNORECASE)
        sugg_tasks_strings = []
        if sugg_tasks_match:
            try:
                sugg_tasks_json_str = sugg_tasks_match.group(1)
                sugg_tasks_strings = json.loads(sugg_tasks_json_str)
                if not isinstance(sugg_tasks_strings, list) or not all(isinstance(s, str) for s in sugg_tasks_strings):
                    log_message(f"Formato JSON inválido para NOVAS_TAREFAS_SUGERIDAS: {sugg_tasks_json_str}", agent_display_name)
                    sugg_tasks_strings = []
                else:
                    log_message(f"Worker: potencial novas tarefas. Parte: {sugg_tasks_strings}", agent_display_name)
            except json.JSONDecodeError as e:
                log_message(f"Erro ao decodificar JSON de NOVAS_TAREFAS_SUGERIDAS: {e}. String: {sugg_tasks_json_str}", agent_display_name)
                sugg_tasks_strings = []
        
        # Remover a seção NOVAS_TAREFAS_SUGERIDAS da resposta principal
        task_res_content = re.sub(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", "", task_res_raw, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # Extrair artefatos de código/markdown
        extracted_artifacts = self._extract_artifacts_from_output(task_res_content)
        
        log_message(f"Resultado da sub-tarefa '{sub_task_description}' (conteúdo principal): {task_res_content[:500]}...", agent_display_name)
        log_message(f"Artefatos extraídos: {len(extracted_artifacts)}", agent_display_name)
        for art in extracted_artifacts: log_message(f"  - {art['filename']} ({art['language']})", agent_display_name)

        # O Worker agora retorna os artefatos extraídos E o texto bruto (sem sugestões)
        # O TaskManager decidirá o que fazer com eles (stage, validar, salvar)
        return {"text_content": task_res_content, "artifacts": extracted_artifacts}, sugg_tasks_strings

    def _extract_artifacts_from_output(self, output_str):
        artifacts = []
        # Regex aprimorada para capturar nome do arquivo e conteúdo
        # Prioriza "Arquivo: nome.ext\n```linguagem..." mas também tenta capturar "```linguagem nome.ext..."
        code_block_pattern = re.compile(
            r"Arquivo:\s*(?P<filename_directive>[^\n`]+)\s*\n"  # "Arquivo: nome.ext"
            r"```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*\n"             # ```linguagem
            r"(?P<content>.*?)\n"                                   # Conteúdo
            r"```",                                                 # ```
            re.DOTALL | re.MULTILINE
        )
        
        # Fallback regex se o nome do arquivo estiver na linha do ```
        code_block_pattern_fallback = re.compile(
            r"```(?P<language_fb>[a-zA-Z0-9_+\-#.]*)\s*(?P<filename_fb>[^\n`]*\.[a-zA-Z0-9_]+)?\s*\n" # ```linguagem nome.ext (opcional)
            r"(?P<content_fb>.*?)\n"                                   # Conteúdo
            r"```",                                                 # ```
            re.DOTALL | re.MULTILINE
        )

        processed_indices = set() # Para evitar processar o mesmo bloco duas vezes

        for match in code_block_pattern.finditer(output_str):
            if match.start() in processed_indices: continue
            processed_indices.add(match.start())

            filename = sanitize_filename(match.group("filename_directive").strip())
            language = match.group("language").strip().lower()
            content = match.group("content").strip()

            if filename and content:
                if content.lower().startswith("arquivo:") and len(content.splitlines()) == 1 :
                    log_message(f"Ignorando artefato que parece ser referência: {filename} com '{content}'", "Worker")
                    continue
                if not content.strip():
                    log_message(f"Ignorando artefato com conteúdo vazio: {filename}", "Worker")
                    continue
                
                # Determinar tipo (code/markdown)
                artifact_type = "code"
                if language == "markdown" or filename.endswith(".md"):
                    artifact_type = "markdown"
                
                artifacts.append({
                    "type": artifact_type, "filename": filename,
                    "content": content, "language": language
                })
            else:
                log_message(f"Bloco de código com 'Arquivo:' mas sem nome/conteúdo válido. Lang: {language}", "Worker")
        
        # Processar blocos com nome de arquivo na linha do ``` (fallback)
        for match in code_block_pattern_fallback.finditer(output_str):
            if match.start() in processed_indices: continue # Já processado pelo padrão principal

            filename_fb = match.group("filename_fb")
            content_fb = match.group("content_fb").strip()
            language_fb = match.group("language_fb").strip().lower()

            if filename_fb and content_fb: # Só considera se tiver nome de arquivo aqui
                filename_fb = sanitize_filename(filename_fb.strip())
                if content_fb.lower().startswith("arquivo:") and len(content_fb.splitlines()) == 1 :
                    log_message(f"Ignorando artefato (fallback) que parece ser referência: {filename_fb} com '{content_fb}'", "Worker")
                    continue
                if not content_fb.strip():
                    log_message(f"Ignorando artefato (fallback) com conteúdo vazio: {filename_fb}", "Worker")
                    continue

                artifact_type = "code"
                if language_fb == "markdown" or filename_fb.endswith(".md"):
                    artifact_type = "markdown"
                
                # Evitar adicionar duplicatas se o padrão principal já pegou
                is_duplicate = False
                for art in artifacts:
                    if art["filename"] == filename_fb and art["content"][:50] == content_fb[:50]: # Checagem simples de duplicata
                        is_duplicate = True
                        break
                if not is_duplicate:
                    artifacts.append({
                        "type": artifact_type, "filename": filename_fb,
                        "content": content_fb, "language": language_fb
                    })
            elif content_fb and not filename_fb and "diff" not in language_fb:
                log_message(f"Bloco de código (fallback) sem nome de arquivo explícito. Lang: {language_fb}. Conteúdo: {content_fb[:100]}...", "Worker")

        return artifacts


class ImageWorker: # Mantida conforme solicitado
    def __init__(self, task_manager, model_name=GEMINI_IMAGE_GENERATION_MODEL_NAME):
        self.task_manager = task_manager
        self.model_name = model_name
        log_message(f"Instância do ImageWorker criada para o modelo Gemini: {self.model_name}", "ImageWorker")
        self.generation_config = generation_config_image_sdk
        log_message(f"ImageWorker usará generation_config: {self.generation_config}", "ImageWorker")

    def generate_image(self, image_prompt, original_task_description="Gerar Imagem"):
        agent_display_name = "ImageWorker"
        print_agent_message(agent_display_name, f"Gerando imagem para o prompt: '{image_prompt[:100]}...'")
        
        # O prompt para o ImageWorker é apenas a descrição da imagem
        prompt_parts_for_api = [image_prompt]

        # Chama a API usando a função genérica, mas com o modelo de imagem e config específica
        response_obj = call_gemini_api_with_retry(
            prompt_parts_for_api, 
            agent_name=agent_display_name, 
            model_name=self.model_name,
            gen_config=self.generation_config 
        )

        if response_obj is None:
            log_message("ImageWorker não obteve resposta da API.", agent_display_name)
            return None, "Falha: Sem resposta da API de imagem."

        try:
            # Acessar a imagem gerada. A estrutura exata pode variar.
            # Para o SDK do Gemini, geralmente é em response.candidates[0].content.parts onde part.image existe
            if response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                for part in response_obj.candidates[0].content.parts:
                    if hasattr(part, 'image') and part.image:
                        # A imagem pode estar em 'part.image.data' (bytes) ou similar
                        # Esta parte é um placeholder e precisa ser ajustada para o SDK real
                        # Se o SDK já retorna um objeto de imagem utilizável (ex: BytesIO ou similar), melhor.
                        # Por agora, vamos assumir que 'part.image' é um objeto que tem 'data' como bytes.
                        
                        image_bytes = part.image.data # Exemplo, pode ser diferente
                        
                        # Salvar a imagem em um arquivo temporário ou diretamente no diretório de saída
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        img_filename_base = sanitize_filename(f"imagem_gerada_{original_task_description[:30]}_{timestamp}")
                        img_filename = f"{img_filename_base}.png" # Assumindo PNG
                        
                        # Salvar no diretório de saída final (OUTPUT_DIRECTORY)
                        # O TaskManager.save_image_artifact fará isso
                        
                        log_message(f"Imagem gerada com sucesso. Pronta para ser salva como '{img_filename}'.", agent_display_name)
                        # Retorna os bytes da imagem e o nome do arquivo para o TaskManager salvar
                        return {"filename": img_filename, "content_bytes": image_bytes, "type": "image"}, None 
                    elif hasattr(part, 'text') and part.text: # Log de qualquer texto retornado
                        log_message(f"ImageWorker recebeu texto da API de imagem: {part.text}", agent_display_name)

            log_message("Nenhuma imagem encontrada na resposta da API.", agent_display_name)
            return None, "Falha: Nenhuma imagem encontrada na resposta."

        except Exception as e:
            log_message(f"Erro ao processar resposta da API de imagem: {e}", agent_display_name)
            log_message(f"Traceback: {traceback.format_exc()}", agent_display_name)
            return None, f"Erro ao processar resposta: {e}"
        
        return None, "Falha: Imagem não gerada por motivo desconhecido."


class Validator:
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Instância do Validator criada.", "Validator")

    def validate_results(self, task_results_history, staged_artifacts, original_goal):
        log_message("Iniciando validação dos resultados e artefatos...", "Validator")
        validation_passed = True
        issues = []
        
        # Critério 1: Verificar se artefatos esperados foram gerados
        # Esta é uma lógica simplificada. Pode ser expandida para verificar
        # se arquivos específicos mencionados na meta ou tarefas foram criados.
        if not staged_artifacts and "criar" in original_goal.lower() or "modificar" in original_goal.lower():
             # Verifica se alguma tarefa tinha "salvar" ou "criar arquivo" e se artefatos correspondentes existem
            expects_files = False
            for task_desc_dict in self.task_manager.current_task_list: # Assumindo que current_task_list tem descrições
                task_desc = list(task_desc_dict.keys())[0] if isinstance(task_desc_dict, dict) else task_desc_dict
                if "salvar" in task_desc.lower() or "criar arquivo" in task_desc.lower() or "gerar o arquivo" in task_desc.lower():
                    expects_files = True
                    break
            if expects_files:
                issues.append("Nenhum artefato de código/markdown foi preparado para salvamento, mas era esperado.")
                validation_passed = False

        # Critério 2: Verificar a qualidade básica dos artefatos preparados
        for filename, artifact_data in staged_artifacts.items():
            content = artifact_data.get("content")
            if not content or not content.strip():
                issues.append(f"Artefato '{filename}' está vazio ou contém apenas espaços em branco.")
                validation_passed = False
            
            # Heurística simples para nomes de arquivo problemáticos (ex: contendo "Arquivo:")
            if "Arquivo:" in filename or "arquivo:" in filename:
                issues.append(f"Nome de arquivo suspeito '{filename}' (contém 'Arquivo:'). Pode ser um erro de parsing.")
                validation_passed = False

            # Heurística para conteúdo que é apenas uma referência
            if isinstance(content, str) and content.lower().startswith("arquivo:") and len(content.splitlines()) <= 2:
                 issues.append(f"Conteúdo do artefato '{filename}' parece ser apenas uma referência de arquivo ('{content[:50]}...'), e não o conteúdo real.")
                 validation_passed = False


        if not validation_passed:
            log_message(f"Validação falhou. Problemas: {'; '.join(issues)}", "Validator")
            # Sugestão de refinamento genérica. Pode ser melhorada com base nos 'issues'.
            suggested_refinements = "Revisar as tarefas de geração e extração de arquivos. Garantir que os nomes dos arquivos sejam corretos e que o conteúdo completo seja fornecido nos formatos especificados."
            return {"status": "failure", "reason": "; ".join(issues), "suggested_refinements": suggested_refinements}

        log_message("Validação bem-sucedida.", "Validator")
        return {"status": "success"}


class TaskManager:
    def __init__(self, initial_goal, uploaded_file_objects=None, uploaded_files_info=None):
        self.goal = initial_goal
        self.uploaded_file_objects = uploaded_file_objects or []
        self.uploaded_files_info = uploaded_files_info or []
        self.current_task_list = []
        self.executed_tasks_results = [] # Histórico de resultados [(desc, res_data), ...]
        self.staged_artifacts = {} # {"filename.ext": {"content": "...", "language": "...", "type": "code/markdown/image"}}
        self.worker = Worker(self)
        self.image_worker = ImageWorker(self) # Mantido
        self.validator = Validator(self)
        log_message("Instância do TaskManager criada.", "TaskManager")

    def decompose_goal(self, goal_to_decompose, previous_plan=None, validation_feedback=None):
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{goal_to_decompose}'")

        files_prompt_part = "Arquivos Complementares: Arquivos complementares carregados (referencie pelo 'Nome de Exibição' ou 'ID do Arquivo'):\n"
        if self.uploaded_files_info:
            for f_meta in self.uploaded_files_info:
                files_prompt_part += f"- Nome: {f_meta['display_name']} (ID: {f_meta['file_id']}, Tipo: {f_meta.get('mime_type', 'N/A')})\n"
        else:
            files_prompt_part += "Nenhum arquivo carregado.\n"

        prompt_parts = [
            "Você é um Gerenciador de Tarefas especialista. Decomponha a meta principal em sub-tarefas sequenciais.",
            f"Meta Principal: \"{goal_to_decompose}\"",
            files_prompt_part,
            # Instruções sobre geração de imagem mantidas
            "Se a meta envolver CRIAÇÃO DE MÚLTIPLAS IMAGENS (ex: \"crie 3 logos\", \"gere 2 variações de um personagem\"), você DEVE:",
            "1.  Criar uma tarefa para gerar a descrição de CADA imagem individualmente. Ex: \"Criar descrição para imagem 1 de [assunto]\".",
            "2.  Seguir CADA tarefa de descrição com uma tarefa \"TASK_GERAR_IMAGEM: [assunto da imagem correspondente]\".",
            "3.  Após TODAS as tarefas de geração de imagem, adicionar UMA tarefa: \"TASK_AVALIAR_IMAGENS: Avaliar as imagens/descrições geradas para [objetivo original] e selecionar as melhores que atendem aos critérios.\"",
            "Se for a CRIAÇÃO DE UMA ÚNICA IMAGEM, use o formato:",
            "1.  \"Criar uma descrição textual detalhada (prompt) para gerar a imagem de [assunto].\"",
            "2.  \"TASK_GERAR_IMAGEM: [assunto da imagem]\"",
            "3.  \"TASK_AVALIAR_IMAGENS: Avaliar a imagem gerada para [objetivo original].\"",
            "Se precisar usar imagem fornecida MAS SEM ENVOLVER CRIAÇÃO DE IMAGENS, use \"TASK_AVALIAR_IMAGENS: Avaliar a imagem fornecida para [objetivo original].\"",
            "Para outras metas, decomponha normalmente. Retorne em JSON array de strings.",
            "Exemplo de CRIAÇÃO de Múltiplas Imagens: [\"Criar descrição para imagem 1 de logo moderno\", \"TASK_GERAR_IMAGEM: Imagem 1 de logo moderno\", \"Criar descrição para imagem 2 de logo vintage\", \"TASK_GERAR_IMAGEM: Imagem 2 de logo vintage\", \"TASK_AVALIAR_IMAGENS: Avaliar os logos gerados para cafeteria e selecionar o melhor.\"]"
        ]
        if previous_plan and validation_feedback:
            prompt_parts.append(f"\nContexto Adicional: O plano anterior falhou na validação.")
            prompt_parts.append(f"Plano Anterior: {json.dumps(previous_plan)}")
            prompt_parts.append(f"Feedback da Validação: {validation_feedback['reason']}")
            prompt_parts.append(f"Sugestões de Refinamento: {validation_feedback['suggested_refinements']}")
            prompt_parts.append("Por favor, gere um NOVO plano de tarefas corrigido e mais detalhado para alcançar a meta original, levando em conta o feedback.")
        
        prompt_parts.append("Sub-tarefas:")
        
        if self.uploaded_file_objects:
            prompt_parts.extend(self.uploaded_file_objects)

        response_str = call_gemini_api_with_retry(prompt_parts, agent_display_name)
        if response_str:
            try:
                # Extrair o JSON do bloco de código, se houver
                match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else: # Se não estiver em bloco de código, assume que a string inteira é o JSON
                    json_str = response_str
                
                tasks = json.loads(json_str)
                if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
                    log_message(f"Tarefas decompostas (strings): {tasks}", agent_display_name)
                    self.current_task_list = tasks
                    return tasks
                else:
                    log_message(f"Resposta da decomposição não é uma lista de strings: {tasks}", agent_display_name)
            except json.JSONDecodeError as e:
                log_message(f"Erro ao decodificar JSON da decomposição: {e}. Resposta: {response_str}", agent_display_name)
        
        print_agent_message(agent_display_name, "Falha ao decompor a meta. Tentando uma abordagem mais simples.")
        # Fallback: se a decomposição falhar, cria uma única tarefa com a meta original
        self.current_task_list = [goal_to_decompose]
        return self.current_task_list

    def process_task_result(self, task_description, task_result_data):
        """Processa o resultado de uma tarefa, incluindo artefatos."""
        self.executed_tasks_results.append({task_description: task_result_data})
        
        if isinstance(task_result_data, dict):
            artifacts = task_result_data.get("artifacts", [])
            for artifact in artifacts:
                filename = artifact.get("filename")
                content = artifact.get("content")
                lang = artifact.get("language")
                art_type = artifact.get("type")

                if filename and content:
                    # Adiciona/sobrescreve no stage. O Validator decidirá o que fazer.
                    self.staged_artifacts[filename] = {
                        "content": content, 
                        "language": lang, 
                        "type": art_type,
                        "source_task": task_description 
                    }
                    log_message(f"Artefato '{filename}' preparado para validação (stage).", "TaskManager")
                elif artifact.get("type") == "image" and filename and artifact.get("content_bytes"):
                     self.staged_artifacts[filename] = {
                        "content_bytes": artifact.get("content_bytes"), 
                        "type": "image",
                        "source_task": task_description 
                    }
                     log_message(f"Artefato de imagem '{filename}' preparado para validação (stage).", "TaskManager")


    def save_final_artifacts(self):
        """Salva os artefatos do stage no diretório de saída final."""
        log_message(f"Salvando {len(self.staged_artifacts)} artefatos finais validados...", "TaskManager")
        saved_files_count = 0
        if not self.staged_artifacts:
            print_agent_message("TaskManager", "Nenhum artefato para salvar.")
            return

        for filename, artifact_data in self.staged_artifacts.items():
            try:
                # Sanitizar o nome do arquivo uma última vez antes de salvar
                final_filename = sanitize_filename(filename)
                if not final_filename:
                    log_message(f"Nome de arquivo inválido ou vazio após sanitização: '{filename}'. Pulando.", "TaskManager")
                    continue

                file_path = os.path.join(OUTPUT_DIRECTORY, final_filename)
                
                if artifact_data.get("type") == "image" and artifact_data.get("content_bytes"):
                    with open(file_path, "wb") as f:
                        f.write(artifact_data["content_bytes"])
                    log_message(f"Imagem final salva: {file_path}", "TaskManager")
                    print_agent_message("TaskManager", f"🖼️ Imagem final salva: {final_filename}")
                    saved_files_count +=1
                elif artifact_data.get("content"): # Para código e markdown
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(artifact_data["content"])
                    log_message(f"Arquivo final salvo: {file_path}", "TaskManager")
                    print_agent_message("TaskManager", f"📄 Arquivo final salvo: {final_filename}")
                    saved_files_count += 1
            except Exception as e:
                log_message(f"Erro ao salvar artefato final '{filename}': {e}", "TaskManager")
                print_agent_message("TaskManager", f"❌ Erro ao salvar artefato final '{filename}'.")
        
        if saved_files_count > 0:
             print_agent_message("TaskManager", f"✅ {saved_files_count} artefato(s) final(is) salvo(s) em: {OUTPUT_DIRECTORY}")
        else:
             print_agent_message("TaskManager", "Nenhum artefato final foi efetivamente salvo.")


    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        log_message(f"Meta inicial: {self.goal}", "TaskManager")

        validation_attempts = 0
        overall_success = False
        
        current_goal_to_decompose = self.goal
        previous_plan_for_replan = None

        while validation_attempts <= MAX_VALIDATION_RETRIES and not overall_success:
            if validation_attempts > 0: # Se não for a primeira tentativa, é um replanejamento
                log_message(f"Tentativa de validação/replanejamento {validation_attempts}/{MAX_VALIDATION_RETRIES}...", "TaskManager")
            
            self.current_task_list = self.decompose_goal(
                current_goal_to_decompose,
                previous_plan=previous_plan_for_replan,
                validation_feedback=self.executed_tasks_results[-1].get("validation_feedback") if self.executed_tasks_results and "validation_feedback" in self.executed_tasks_results[-1] else None
            )

            if not self.current_task_list:
                print_agent_message("TaskManager", "Não foi possível decompor a meta em tarefas. Encerrando.")
                log_message("Falha na decomposição da meta. Encerrando.", "TaskManager")
                return

            print_agent_message("TaskManager", "--- PLANO DE TAREFAS INICIAL ---")
            for i, task_desc in enumerate(self.current_task_list):
                print(f"  {i+1}. {task_desc}")
            
            print_user_message("Aprova este plano? (s/n)")
            if input("➡️ ").strip().lower() != 's':
                print_agent_message("TaskManager", "Plano não aprovado. Encerrando.")
                log_message("Plano não aprovado pelo usuário.", "UsuárioInput")
                return

            log_message("Plano aprovado pelo usuário.", "UsuárioInput")
            
            # Limpar resultados e artefatos de tentativas anteriores antes de executar novo plano
            self.executed_tasks_results = [] 
            self.staged_artifacts = {}

            for i, task_description_str in enumerate(self.current_task_list):
                print_agent_message("TaskManager", f"Próxima tarefa ({i+1}/{len(self.current_task_list)}): {task_description_str}")
                
                task_result_data = {}
                suggested_new_tasks = []

                if task_description_str.startswith("TASK_GERAR_IMAGEM:"):
                    image_prompt = task_description_str.replace("TASK_GERAR_IMAGEM:", "").strip()
                    image_artifact, error_msg = self.image_worker.generate_image(image_prompt, original_task_description=task_description_str)
                    if image_artifact:
                        task_result_data = {"image_artifact": image_artifact, "text_content": f"Imagem '{image_artifact['filename']}' gerada."}
                        # Adiciona a imagem diretamente aos artefatos preparados
                        self.staged_artifacts[image_artifact['filename']] = {
                            "content_bytes": image_artifact["content_bytes"], 
                            "type": "image",
                            "source_task": task_description_str
                        }
                        log_message(f"Imagem '{image_artifact['filename']}' preparada para validação.", "TaskManager")
                    else:
                        task_result_data = {"text_content": f"Falha ao gerar imagem: {error_msg}"}
                
                elif task_description_str.startswith("TASK_AVALIAR_IMAGENS:"):
                    # Lógica de avaliação de imagem (pode ser uma chamada LLM ou heurística)
                    # Por agora, vamos assumir que a avaliação é positiva se imagens foram geradas
                    eval_prompt = f"Avaliar as imagens geradas (se houver) para o objetivo: {self.goal}. As imagens nos artefatos preparados são adequadas?"
                    # Esta tarefa pode precisar acessar self.staged_artifacts que contêm imagens
                    eval_result = call_gemini_api_with_retry([eval_prompt] + self.uploaded_file_objects, "Validator(AvaliaçãoImagem)")
                    task_result_data = {"text_content": f"Resultado da avaliação de imagens: {eval_result or 'N/A'}"}
                
                else: # Tarefa padrão para o Worker
                    task_result_data, suggested_new_tasks = self.worker.execute_task(
                        task_description_str,
                        self.executed_tasks_results,
                        self.uploaded_files_info,
                        self.goal
                    )
                
                self.process_task_result(task_description_str, task_result_data)
                log_message(f"Tarefa '{task_description_str}' concluída.", "TaskManager")

                if suggested_new_tasks: # Adiciona novas tarefas sugeridas ao final da lista atual
                    print_agent_message("TaskManager", f"Novas tarefas sugeridas pelo Worker: {suggested_new_tasks}")
                    self.current_task_list.extend(suggested_new_tasks)
                    log_message(f"Lista de tarefas atualizada com sugestões: {self.current_task_list}", "TaskManager")
            
            # Após todas as tarefas do plano atual, validar
            validation_result = self.validator.validate_results(self.executed_tasks_results, self.staged_artifacts, self.goal)

            if validation_result["status"] == "success":
                print_agent_message("TaskManager", "Validação dos artefatos bem-sucedida!")
                self.save_final_artifacts() # Salva os artefatos do stage
                overall_success = True
                break # Sai do loop de validação/replanejamento
            else:
                print_agent_message("TaskManager", f"Validação falhou: {validation_result['reason']}")
                validation_attempts += 1
                if validation_attempts <= MAX_VALIDATION_RETRIES:
                    print_agent_message("TaskManager", "Tentando replanejar as tarefas...")
                    current_goal_to_decompose = self.goal # Pode ser refinado com base no feedback
                    previous_plan_for_replan = list(self.current_task_list) # Salva o plano que falhou
                    # Adiciona o feedback da validação ao histórico para o próximo ciclo de decomposição
                    self.executed_tasks_results.append({"validation_feedback": validation_result})
                else:
                    print_agent_message("TaskManager", "Número máximo de tentativas de validação atingido. Encerrando.")
                    log_message("Máximo de tentativas de validação atingido.", "TaskManager")
                    break
        
        if overall_success:
            print_agent_message("TaskManager", "Fluxo de trabalho concluído com sucesso!")
        else:
            print_agent_message("TaskManager", "Fluxo de trabalho concluído com falhas de validação não resolvidas.")
        
        # Limpar arquivos temporários (se houver) - não implementado neste exemplo,
        # pois os artefatos são mantidos em self.staged_artifacts em memória.

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.0" 
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
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
        task_manager = TaskManager(initial_goal_input, uploaded_files, uploaded_files_meta)
        task_manager.run_workflow()

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"\n--- Execução ({SCRIPT_VERSION}) Finalizada ---")



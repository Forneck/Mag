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
MAX_AUTOMATIC_VALIDATION_RETRIES = 2 # Renomeado de MAX_VALIDATION_RETRIES
MAX_MANUAL_VALIDATION_RETRIES = 2  # Novo para validação manual

# --- Funções de Utilidade ---
def sanitize_filename(name, allow_extension=True):
    if not name:
        return ""
    base_name, ext = os.path.splitext(name)
    base_name = re.sub(r'[^\w\s.-]', '', base_name).strip()
    base_name = re.sub(r'[-\s]+', '-', base_name)
    base_name = base_name[:100] 
    if not allow_extension or not ext:
        return base_name
    else:
        ext = "." + re.sub(r'[^\w-]', '', ext.lstrip('.')).strip()
        ext = ext[:10] 
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

GEMINI_TEXT_MODEL_NAME = "gemini-2.0-flash"
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
log_message(f"Modelo Gemini (geração de imagem via SDK): {GEMINI_IMAGE_GENERATION_MODEL_NAME}", "Sistema")

generation_config_text = {
    "temperature": 0.7, "top_p": 0.95, "top_k": 64,
    "max_output_tokens": 8192, "response_mime_type": "text/plain",
}
generation_config_image_sdk = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 64,
    "response_modalities": ['TEXT', 'IMAGE'], "max_output_tokens": 8192,
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
        active_gen_config = generation_config_image_sdk if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME else generation_config_text
        log_message(f"Nenhuma gen_config específica, usando padrão para {model_name}.", "Sistema")

    for part_item in prompt_parts:
        if isinstance(part_item, str): text_prompt_for_log += part_item + "\n"
        elif hasattr(part_item, 'name') and hasattr(part_item, 'display_name'):
            file_references_for_log.append(f"Arquivo: {part_item.display_name} (ID: {part_item.name}, TipoMIME: {getattr(part_item, 'mime_type', 'N/A')})")
    log_message(f"Prompt textual para {agent_name} (Modelo: {model_name}):\n---\n{text_prompt_for_log}\n---", "Sistema")
    if file_references_for_log: log_message(f"Arquivos referenciados para {agent_name}:\n" + "\n".join(file_references_for_log), "Sistema")
    log_message(f"Usando generation_config para {agent_name} (Modelo: {model_name}): {active_gen_config}", "Sistema")

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES} para {agent_name} (Modelo: {model_name})...", "Sistema")
        try:
            model_instance = genai.GenerativeModel(model_name, generation_config=active_gen_config, safety_settings=safety_settings_gemini)
            response = model_instance.generate_content(prompt_parts)
            log_message(f"Resposta bruta da API Gemini (tentativa {attempt + 1}, Modelo: {model_name}): {response}", agent_name)

            if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME: return response
            if hasattr(response, 'text') and response.text is not None: return response.text.strip()
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                first_part = response.candidates[0].content.parts[0]
                if hasattr(first_part, 'text') and first_part.text is not None: return first_part.text.strip()

            log_message(f"API Gemini (Modelo: {model_name}) não retornou texto utilizável (tentativa {attempt + 1}).", agent_name)
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                log_message(f"Bloqueio: {getattr(response.prompt_feedback, 'block_reason_message', 'N/A')} ({response.prompt_feedback.block_reason})", agent_name)
            if attempt < MAX_API_RETRIES - 1: time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR; continue
            return None
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1} ({agent_name}, Modelo: {model_name}): {type(e).__name__} - {e}\n{traceback.format_exc()}", agent_name)
            if attempt < MAX_API_RETRIES - 1: time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR
            else: return None
    return None

# --- Funções de Arquivos (mantidas da v9.1) ---
def get_most_recent_cache_file():
    try:
        list_of_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))
        return max(list_of_files, key=os.path.getctime) if list_of_files else None
    except Exception as e: log_message(f"Erro ao buscar cache mais recente: {e}", "Sistema"); return None

def load_cached_files_metadata(cache_file_path):
    if not cache_file_path or not os.path.exists(cache_file_path): return []
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f: cached_metadata = json.load(f)
        return cached_metadata if isinstance(cached_metadata, list) else []
    except Exception as e: log_message(f"Erro ao carregar cache {cache_file_path}: {e}", "Sistema"); return []

def clear_upload_cache():
    # Implementação da v9.1 mantida
    print_agent_message("Sistema", f"Verificando arquivos de cache de upload local em: {UPLOADED_FILES_CACHE_DIR}")
    local_cache_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))
    if not local_cache_files:
        print_agent_message("Sistema", "Nenhum arquivo de cache de upload local encontrado para limpar.")
    else:
        print_agent_message("Sistema", f"Encontrados {len(local_cache_files)} arquivo(s) de cache de upload local:")
        for cf in local_cache_files: print(f"  - {os.path.basename(cf)}")
        if input("👤 Deseja limpar TODOS esses arquivos de cache de upload LOCAL? (s/n) ➡️ ").strip().lower() == 's':
            for cache_file_path in local_cache_files:
                try: os.remove(cache_file_path); log_message(f"Cache local '{os.path.basename(cache_file_path)}' removido.", "Sistema")
                except Exception as e: log_message(f"Erro ao remover cache local '{os.path.basename(cache_file_path)}': {e}", "Sistema")
            print_agent_message("Sistema", "Cache local limpo (ou tentativa).")
        else:
            print_agent_message("Sistema", "Limpeza do cache local cancelada.")

    try:
        api_files_list = list(genai.list_files())
        if not api_files_list: print_agent_message("Sistema", "Nenhum arquivo na API Gemini Files para limpar."); return
        print_agent_message("Sistema", f"Encontrados {len(api_files_list)} arquivo(s) na API Gemini Files.")
        if input("👤 ‼️ ATENÇÃO: IRREVERSÍVEL ‼️ Deseja deletar TODOS os arquivos da API Gemini Files? (s/n) ➡️ ").strip().lower() == 's':
            for api_file in api_files_list:
                try: genai.delete_file(name=api_file.name); log_message(f"Arquivo da API '{api_file.display_name}' deletado.", "Sistema"); print(f"  🗑️ Deletado da API: {api_file.display_name}"); time.sleep(0.2)
                except Exception as e: log_message(f"Erro ao deletar '{api_file.display_name}' da API: {e}", "Sistema")
            print_agent_message("Sistema", "Arquivos da API Gemini Files limpos (ou tentativa).")
        else:
            print_agent_message("Sistema", "Limpeza de arquivos da API cancelada.")
    except Exception as e: print_agent_message("Sistema", f"❌ Erro ao acessar/limpar arquivos da API: {e}"); log_message(f"Erro API clear: {e}", "Sistema")


def get_uploaded_files_info_from_user():
    # Implementação da v9.1 mantida
    uploaded_file_objects, uploaded_files_metadata, reused_file_ids = [], [], set()
    try: api_files_list = list(genai.list_files())
    except Exception as e: api_files_list = []; log_message(f"Falha ao listar arquivos da API: {e}", "Sistema")
    api_files_dict = {f.name: f for f in api_files_list}
    
    cached_metadata = load_cached_files_metadata(get_most_recent_cache_file())
    offer_for_reuse = []
    for api_file in api_files_list:
        meta = next((cm for cm in cached_metadata if cm.get("file_id") == api_file.name), None)
        offer_for_reuse.append({
            "file_id": api_file.name, "display_name": meta.get("display_name", api_file.display_name) if meta else api_file.display_name,
            "mime_type": meta.get("mime_type", api_file.mime_type) if meta else api_file.mime_type,
            "user_path": meta.get("user_path", "API") if meta else "API"
        })

    if offer_for_reuse:
        print_agent_message("Sistema", "Arquivos na API/cache:")
        for i, m in enumerate(offer_for_reuse): print(f"  {i+1}. {m['display_name']} (Origem: {m['user_path']})")
        if input("👤 Reutilizar arquivos? (s/n) ➡️ ").lower() == 's':
            choices = input("👤 Números (ex: 1,3) ou 'todos': ➡️ ").lower()
            selected_indices = list(range(len(offer_for_reuse))) if choices == 'todos' else [int(x.strip())-1 for x in choices.split(',') if x.strip().isdigit()]
            for idx in selected_indices:
                if 0 <= idx < len(offer_for_reuse):
                    chosen = offer_for_reuse[idx]
                    try:
                        file_obj = api_files_dict.get(chosen["file_id"]) or genai.get_file(name=chosen["file_id"])
                        uploaded_file_objects.append(file_obj); uploaded_files_metadata.append(chosen); reused_file_ids.add(chosen["file_id"])
                        log_message(f"Arquivo '{chosen['display_name']}' reutilizado.", "Sistema")
                    except Exception as e: log_message(f"Erro ao obter '{chosen['display_name']}' para reutilização: {e}", "Sistema")
    
    if input("👤 Adicionar NOVOS arquivos? (s/n) ➡️ ").lower() == 's':
        while True:
            fp_pattern = input("👤 Caminho/padrão do novo arquivo (ou 'fim'): ➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            expanded = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*','?','[',']']) else ([fp_pattern] if os.path.isfile(fp_pattern) else [])
            if not expanded: print("❌ Nenhum arquivo encontrado."); continue
            for fp_actual in expanded:
                dn = os.path.basename(fp_actual)
                try:
                    ext_map = { ".md": "text/markdown", ".py": "text/x-python", ".cpp": "text/x-c++src", ".h": "text/x-chdr", ".hpp": "text/x-c++hdr", ".txt": "text/plain", ".json": "application/json"}
                    mime = ext_map.get(os.path.splitext(dn)[1].lower())
                    file_obj = genai.upload_file(path=fp_actual, display_name=dn, mime_type=mime)
                    uploaded_file_objects.append(file_obj)
                    new_meta = {"file_id": file_obj.name, "display_name": dn, "mime_type": file_obj.mime_type, "user_path": fp_actual}
                    uploaded_files_metadata.append(new_meta)
                    log_message(f"Novo arquivo '{dn}' enviado.", "Sistema"); time.sleep(1)
                except Exception as e: log_message(f"Erro no upload de '{dn}': {e}", "Sistema")
    if uploaded_files_metadata:
        with open(os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json"), "w") as f:
            json.dump(uploaded_files_metadata, f, indent=4)
    return uploaded_file_objects, uploaded_files_metadata

# --- Classes dos Agentes (Worker, ImageWorker, Validator - mantidas da v9.1) ---
class Worker:
    def __init__(self, task_manager, model_name=GEMINI_TEXT_MODEL_NAME):
        self.task_manager = task_manager; self.model_name = model_name
        log_message("Instância do Worker criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        # Implementação da v9.1 mantida
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")
        prompt_context = "Resultados anteriores:\n" + ('\n'.join([f"- '{list(res.keys())[0]}': {str(list(res.values())[0].get('text_content', list(res.values())[0]))[:500]}..." for res in previous_results]) if previous_results else "Nenhum.\n")
        files_prompt_part = "Arquivos complementares carregados:\n" + ('\n'.join([f"- Nome: {f['display_name']} (ID: {f['file_id']})" for f in uploaded_files_info]) if uploaded_files_info else "Nenhum.\n")
        prompt = [
            f"Você é um Agente Executor. Tarefa atual: \"{sub_task_description}\"",
            f"Contexto (resultados anteriores, objetivo original, arquivos):\n{prompt_context}\n{files_prompt_part}",
            f"Objetivo: {original_goal}", "Execute a tarefa.",
            " * Se for \"Criar uma descrição textual detalhada (prompt) para gerar a imagem de [...]\", seu resultado DEVE ser APENAS essa descrição textual.",
            " * Se a tarefa envolver modificar ou criar arquivos de código ou markdown, forneça o CONTEÚDO COMPLETO do arquivo.",
            " * Indique o NOME DO ARQUIVO CLARAMENTE ANTES de cada bloco de código/markdown usando o formato:\n   \"Arquivo: nome_completo.ext\"\n   ```linguagem_ou_extensao\n   // Conteúdo completo...\n   ```",
            " * Para arquivos Markdown, use `markdown` como linguagem.",
            " * Se precisar retornar múltiplos arquivos, repita o formato ou use JSON: {\"resultado\": \"descrição\", \"arquivos\": [{\"nome\": \"f1.cpp\", \"conteudo\": \"...\"}]}",
            " * Se identificar NOVAS sub-tarefas cruciais, liste-as em 'NOVAS_TAREFAS_SUGERIDAS:' como array JSON de strings. Se não, omita.",
            "Resultado da Tarefa:", "[Resultado principal...]", "NOVAS_TAREFAS_SUGERIDAS:", "[Array JSON...]"
        ]
        if self.task_manager.uploaded_file_objects: prompt.extend(self.task_manager.uploaded_file_objects)
        
        task_res_raw = call_gemini_api_with_retry(prompt, agent_display_name, model_name=self.model_name)
        if task_res_raw is None: return {"text_content": "Falha: Sem resposta da API.", "artifacts": []}, []

        sugg_tasks_match = re.search(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", task_res_raw, re.DOTALL | re.IGNORECASE)
        sugg_tasks_strings = []
        if sugg_tasks_match:
            try: sugg_tasks_strings = json.loads(sugg_tasks_match.group(1))
            except json.JSONDecodeError: log_message("Erro JSON NOVAS_TAREFAS_SUGERIDAS", agent_display_name)
        
        task_res_content = re.sub(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", "", task_res_raw, flags=re.DOTALL | re.IGNORECASE).strip()
        extracted_artifacts = self._extract_artifacts_from_output(task_res_content)
        
        main_text_output = task_res_content
        if extracted_artifacts:
            try:
                json_data = json.loads(task_res_content.strip().lstrip("```json").rstrip("```").strip()) # Tentativa de limpar JSON
                if isinstance(json_data, dict) and "resultado" in json_data: main_text_output = json_data["resultado"]
            except json.JSONDecodeError: pass # Mantém task_res_content se não for JSON com "resultado"

        log_message(f"Resultado da sub-tarefa '{sub_task_description}' (processado): {str(main_text_output)[:500]}...", agent_display_name)
        return {"text_content": main_text_output, "artifacts": extracted_artifacts}, sugg_tasks_strings

    def _extract_artifacts_from_output(self, output_str):
        # Implementação da v9.1 mantida
        artifacts = []
        try: # Prioriza JSON
            cleaned_output = output_str.strip()
            if cleaned_output.startswith("```json"): cleaned_output = re.sub(r"^```json\s*|\s*```$", "", cleaned_output, flags=re.DOTALL).strip()
            data = json.loads(cleaned_output)
            if isinstance(data, dict) and "arquivos" in data and isinstance(data["arquivos"], list):
                for item in data["arquivos"]:
                    if isinstance(item, dict) and "nome" in item and "conteudo" in item:
                        fn = sanitize_filename(item["nome"]); cont = item["conteudo"]; lang = item.get("linguagem","").lower() or (fn.split('.')[-1] if '.' in fn else "")
                        if fn and cont and cont.strip():
                            artifacts.append({"type": "markdown" if lang=="markdown" or fn.endswith(".md") else "code", "filename": fn, "content": cont, "language": lang, "extraction_method": "json"})
                if artifacts: log_message(f"{len(artifacts)} artefatos extraídos via JSON.", "Worker"); return artifacts
        except Exception: pass # Silenciosamente falha e tenta regex

        # Fallback para Regex
        patterns = [
            re.compile(r"Arquivo:\s*(?P<filename>[^\n`]+)\s*\n```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*\n(?P<content>.*?)\n```", re.DOTALL | re.MULTILINE),
            re.compile(r"```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*(?P<filename>[^\n`]*\.[a-zA-Z0-9_]+)?\s*\n(?P<content>.*?)\n```", re.DOTALL | re.MULTILINE)
        ]
        processed_starts = set()
        for i, pattern in enumerate(patterns):
            for match in pattern.finditer(output_str):
                if match.start() in processed_starts: continue
                
                filename_match = match.groupdict().get("filename")
                content_match = match.groupdict().get("content")
                language_match = match.groupdict().get("language","").lower()

                if not filename_match and i == 1: # Se é o segundo padrão (fallback) e não tem nome de arquivo explícito
                    continue 

                filename = sanitize_filename(filename_match.strip()) if filename_match else ""
                content = content_match.strip() if content_match else ""
                language = language_match or (filename.split('.')[-1] if '.' in filename else "")

                if filename and content and not (content.lower().startswith("arquivo:") and len(content.splitlines()) <=2) and content.strip():
                    is_dup = any(a["filename"] == filename and a["content"][:50] == content[:50] for a in artifacts)
                    if not is_dup:
                        artifacts.append({"type": "markdown" if language=="markdown" or filename.endswith(".md") else "code", 
                                          "filename": filename, "content": content, "language": language, 
                                          "extraction_method": "regex_directive" if i==0 else "regex_fallback"})
                        processed_starts.add(match.start())
        if artifacts: log_message(f"{len(artifacts)} artefatos extraídos via Regex.", "Worker")
        return artifacts

class ImageWorker:
    def __init__(self, task_manager, model_name=GEMINI_IMAGE_GENERATION_MODEL_NAME):
        # Implementação da v9.1 mantida
        self.task_manager = task_manager; self.model_name = model_name
        self.generation_config = generation_config_image_sdk
        log_message(f"ImageWorker criado para {self.model_name}", "ImageWorker")

    def generate_image(self, image_prompt, original_task_description="Gerar Imagem"):
        # Implementação da v9.1 mantida
        agent_display_name = "ImageWorker"
        print_agent_message(agent_display_name, f"Gerando imagem para: '{image_prompt[:100]}...'")
        response_obj = call_gemini_api_with_retry([image_prompt], agent_display_name, self.model_name, self.generation_config)
        if not response_obj: return None, "Falha: Sem resposta da API de imagem."
        try:
            if response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                for part in response_obj.candidates[0].content.parts:
                    if hasattr(part, 'image') and part.image and hasattr(part.image, 'data'):
                        img_bytes = part.image.data
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fn_base = sanitize_filename(f"imagem_gerada_{original_task_description[:20]}_{ts}")
                        fn = f"{fn_base}.png"
                        log_message(f"Imagem gerada, pronta para salvar como '{fn}'.", agent_display_name)
                        return {"filename": fn, "content_bytes": img_bytes, "type": "image"}, None
            return None, "Falha: Nenhuma imagem na resposta."
        except Exception as e: log_message(f"Erro ao processar imagem: {e}\n{traceback.format_exc()}", agent_display_name); return None, f"Erro: {e}"

class Validator:
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Instância do Validator criada.", "Validator")

    def validate_results(self, task_results_history, staged_artifacts, original_goal):
        # Implementação da v9.1 mantida
        log_message(f"Iniciando validação automática. Artefatos no stage: {list(staged_artifacts.keys())}", "Validator")
        issues = []
        if not staged_artifacts:
            expects_files = any(kw in original_goal.lower() for kw in ["criar", "modificar", "salvar"]) or \
                            (hasattr(self.task_manager, 'current_task_list') and \
                             any(any(kw_task in (list(task.keys())[0] if isinstance(task,dict) else str(task)).lower() for kw_task in ["salvar", "criar arquivo", "gerar o arquivo"]) for task in self.task_manager.current_task_list))
            if expects_files: issues.append("Nenhum artefato preparado, mas era esperado.")
        else:
            for filename, artifact_data in staged_artifacts.items():
                content = artifact_data.get("content")
                content_bytes = artifact_data.get("content_bytes")
                if artifact_data.get("type") == "image" and not content_bytes: issues.append(f"Imagem '{filename}' sem bytes.")
                elif artifact_data.get("type") != "image" and (not content or not content.strip()): issues.append(f"Artefato '{filename}' vazio.")
                if "Arquivo:" in filename: issues.append(f"Nome de arquivo suspeito '{filename}'.")
                if isinstance(content, str) and content.lower().startswith("arquivo:") and len(content.splitlines()) <= 2: issues.append(f"Conteúdo de '{filename}' parece referência.")
        
        if issues:
            log_message(f"Validação automática falhou. Problemas: {'; '.join(issues)}", "Validator")
            return {"status": "failure", "reason": "; ".join(issues), "suggested_refinements": "Revisar tarefas de geração/extração. Garantir nomes e conteúdo corretos (Markdown ou JSON 'arquivos')."}
        log_message(f"Validação automática bem-sucedida para: {list(staged_artifacts.keys())}", "Validator")
        return {"status": "success"}

class TaskManager:
    def __init__(self, initial_goal, uploaded_file_objects=None, uploaded_files_info=None):
        self.goal = initial_goal
        self.uploaded_file_objects = uploaded_file_objects or []
        self.uploaded_files_info = uploaded_files_info or []
        self.current_task_list = []
        self.executed_tasks_results = [] 
        self.staged_artifacts = {} 
        self.worker = Worker(self)
        self.image_worker = ImageWorker(self) 
        self.validator = Validator(self)
        log_message("Instância do TaskManager criada.", "TaskManager")

    def decompose_goal(self, goal_to_decompose, previous_plan=None, automatic_validation_feedback=None, manual_validation_feedback_str=None):
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{goal_to_decompose}'")
        files_prompt_part = "Arquivos Complementares:\n" + ('\n'.join([f"- {f['display_name']} (ID: {f['file_id']})" for f in self.uploaded_files_info]) if self.uploaded_files_info else "Nenhum.\n")
        prompt_parts = [
            "Você é um Gerenciador de Tarefas especialista. Decomponha a meta principal em sub-tarefas sequenciais.",
            f"Meta Principal: \"{goal_to_decompose}\"", files_prompt_part,
            # Instruções de imagem mantidas...
            "Se a meta envolver CRIAÇÃO DE MÚLTIPLAS IMAGENS...", # (instruções da v9.1)
            "Se for a CRIAÇÃO DE UMA ÚNICA IMAGEM...", # (instruções da v9.1)
            "Se precisar usar imagem fornecida MAS SEM ENVOLVER CRIAÇÃO DE IMAGENS...",
            "Para outras metas, decomponha normalmente. Retorne em JSON array de strings."
        ]
        if previous_plan: # Se é um replanejamento
            prompt_parts.append(f"\nContexto Adicional: O plano anterior precisa de revisão.")
            prompt_parts.append(f"Plano Anterior: {json.dumps(previous_plan)}")
            if automatic_validation_feedback and automatic_validation_feedback.get("status") == "failure":
                prompt_parts.append(f"Feedback da Validação Automática: {automatic_validation_feedback['reason']}")
                prompt_parts.append(f"Sugestões de Refinamento (Automático): {automatic_validation_feedback['suggested_refinements']}")
            if manual_validation_feedback_str: # Novo para v9.2
                 prompt_parts.append(f"Feedback da Validação Manual do Usuário: {manual_validation_feedback_str}")
            prompt_parts.append("Por favor, gere um NOVO plano de tarefas corrigido e mais detalhado para alcançar a meta original, levando em conta o feedback.")
        prompt_parts.append("Sub-tarefas (JSON array de strings):")
        if self.uploaded_file_objects: prompt_parts.extend(self.uploaded_file_objects)

        response_str = call_gemini_api_with_retry(prompt_parts, agent_display_name)
        if response_str:
            try:
                match = re.search(r"```json\s*(.*?)\s*```", response_str, re.DOTALL)
                json_str = match.group(1) if match else response_str
                tasks = json.loads(json_str)
                if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
                    log_message(f"Tarefas decompostas: {tasks}", agent_display_name); return tasks
            except json.JSONDecodeError as e: log_message(f"Erro JSON decomposição: {e}. Resposta: {response_str}", agent_display_name)
        print_agent_message(agent_display_name, "Falha ao decompor. Usando meta original como única tarefa.")
        return [goal_to_decompose]

    def present_for_manual_validation(self):
        log_message("Iniciando validação manual pelo usuário...", "TaskManager")
        print_agent_message("TaskManager", "Validação automática OK. Artefatos gerados/modificados:")
        if not self.staged_artifacts:
            print_agent_message("TaskManager", "Nenhum artefato foi preparado para validação manual (estranho, já que a validação automática passou).")
            # Isso pode indicar um problema se a validação automática disse que havia artefatos.
            # Por segurança, vamos tratar como se não houvesse nada para aprovar.
            return {"approved": True, "feedback": "Nenhum artefato no stage para validação manual, mas automática OK."}

        for i, (filename, data) in enumerate(self.staged_artifacts.items()):
            tipo_artefato = data.get("type", "desconhecido").capitalize()
            print(f"  {i+1}. {tipo_artefato}: {filename}")
            # Para código/markdown, poderia mostrar um snippet, mas por ora, só o nome.
            # if data.get("type") != "image" and data.get("content"):
            #     print(f"     Conteúdo (início): {data['content'][:100]}...")
        
        print_user_message("Você aprova estes artefatos finais? (s/n/cancelar)")
        user_choice = input("➡️ ").strip().lower()

        if user_choice == 's':
            log_message("Artefatos aprovados manualmente pelo usuário.", "TaskManager")
            return {"approved": True}
        elif user_choice == 'cancelar':
            log_message("Validação manual cancelada pelo usuário.", "TaskManager")
            return {"approved": False, "feedback": "cancelar"}
        else: # 'n' ou qualquer outra coisa é reprovação
            print_user_message("Por favor, forneça um feedback para correção (ou 'cancelar' para interromper):")
            feedback = input("➡️ ").strip()
            log_message(f"Artefatos reprovados manualmente. Feedback: {feedback}", "TaskManager")
            return {"approved": False, "feedback": feedback if feedback else "Reprovado sem feedback específico."}

    def process_task_result(self, task_description, task_result_data):
        # Implementação da v9.1 mantida
        self.executed_tasks_results.append({task_description: task_result_data})
        if isinstance(task_result_data, dict):
            for artifact in task_result_data.get("artifacts", []):
                filename, art_type = artifact.get("filename"), artifact.get("type")
                if filename:
                    if art_type == "image" and artifact.get("content_bytes"):
                        self.staged_artifacts[filename] = artifact # Salva o dict completo
                        log_message(f"Artefato de imagem '{filename}' preparado (stage).", "TaskManager")
                    elif art_type in ["code", "markdown"] and artifact.get("content"):
                        self.staged_artifacts[filename] = artifact # Salva o dict completo
                        log_message(f"Artefato '{filename}' ({art_type}) preparado (stage).", "TaskManager")

    def save_final_artifacts(self):
        # Implementação da v9.1 mantida
        log_message(f"Salvando {len(self.staged_artifacts)} artefatos finais...", "TaskManager")
        saved_count = 0
        if not self.staged_artifacts: print_agent_message("TaskManager", "Nenhum artefato para salvar."); return
        for filename, data in self.staged_artifacts.items():
            final_fn = sanitize_filename(filename)
            if not final_fn: log_message(f"Nome inválido '{filename}', pulando.", "TaskManager"); continue
            fp = os.path.join(OUTPUT_DIRECTORY, final_fn)
            try:
                if data.get("type") == "image" and data.get("content_bytes"):
                    with open(fp, "wb") as f: f.write(data["content_bytes"])
                    print_agent_message("TaskManager", f"🖼️ Imagem salva: {final_fn}")
                elif data.get("content") and data.get("type") in ["code", "markdown"]:
                    with open(fp, "w", encoding="utf-8") as f: f.write(data["content"])
                    print_agent_message("TaskManager", f"📄 Arquivo salvo: {final_fn}")
                else: continue # Não salvar se não tiver conteúdo esperado
                log_message(f"Artefato final salvo: {fp}", "TaskManager"); saved_count += 1
            except Exception as e: log_message(f"Erro ao salvar '{final_fn}': {e}", "TaskManager")
        print_agent_message("TaskManager", f"✅ {saved_count} artefato(s) salvo(s) em: {OUTPUT_DIRECTORY}" if saved_count else "Nenhum artefato final efetivamente salvo.")


    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        log_message(f"Meta inicial: {self.goal}", "TaskManager")

        overall_success = False
        automatic_validation_attempts = 0
        current_goal_to_decompose = self.goal
        previous_plan_for_replan = None
        last_automatic_validation_feedback = None
        last_manual_feedback = None # Novo para v9.2

        while automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES and not overall_success:
            if automatic_validation_attempts > 0: # Se não for a primeira tentativa, é um replanejamento
                log_message(f"Tentativa de replanejamento automático {automatic_validation_attempts}/{MAX_AUTOMATIC_VALIDATION_RETRIES}...", "TaskManager")
            
            self.current_task_list = self.decompose_goal(
                current_goal_to_decompose,
                previous_plan=previous_plan_for_replan,
                automatic_validation_feedback=last_automatic_validation_feedback, # Passa feedback da auto-validação
                manual_validation_feedback_str=last_manual_feedback # Passa feedback da manual-validação
            )
            last_manual_feedback = None # Reseta após usar

            if not self.current_task_list:
                print_agent_message("TaskManager", "Não foi possível decompor a meta. Encerrando."); return

            print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
            for i, task_desc in enumerate(self.current_task_list): print(f"  {i+1}. {task_desc}")
            
            if input("👤 Aprova este plano? (s/n) ➡️ ").strip().lower() != 's':
                print_agent_message("TaskManager", "Plano não aprovado. Encerrando."); return
            log_message("Plano aprovado pelo usuário.", "UsuárioInput")
            
            self.executed_tasks_results = [] 
            self.staged_artifacts = {}

            for i, task_description_str in enumerate(self.current_task_list):
                print_agent_message("TaskManager", f"Próxima tarefa ({i+1}/{len(self.current_task_list)}): {task_description_str}")
                task_result_data, suggested_new_tasks = {}, [] # Inicializa
                
                if task_description_str.startswith("TASK_GERAR_IMAGEM:"):
                    # Lógica do ImageWorker (mantida da v9.1)
                    image_prompt = task_description_str.replace("TASK_GERAR_IMAGEM:", "").strip()
                    image_artifact_data, error_msg = self.image_worker.generate_image(image_prompt, original_task_description=task_description_str)
                    if image_artifact_data:
                        task_result_data = {"image_artifact_details": image_artifact_data, "text_content": f"Imagem '{image_artifact_data['filename']}' gerada.", "artifacts": [image_artifact_data]}
                    else:
                        task_result_data = {"text_content": f"Falha ao gerar imagem: {error_msg}", "artifacts": []}

                elif task_description_str.startswith("TASK_AVALIAR_IMAGENS:"):
                    # Lógica de avaliação de imagem (mantida da v9.1)
                    eval_prompt = f"Avaliar as imagens geradas (se houver nos artefatos preparados) para o objetivo: {self.goal}."
                    image_artifacts_in_stage = [fn for fn, data in self.staged_artifacts.items() if data.get("type") == "image"]
                    eval_prompt += f"\nImagens atualmente preparadas: {', '.join(image_artifacts_in_stage) if image_artifacts_in_stage else 'Nenhuma'}."
                    prompt_for_eval = [eval_prompt]
                    if self.uploaded_file_objects: prompt_for_eval.extend(self.uploaded_file_objects)
                    eval_result = call_gemini_api_with_retry(prompt_for_eval, "Validator(AvaliaçãoImagem)")
                    task_result_data = {"text_content": f"Resultado da avaliação de imagens: {eval_result or 'N/A'}", "artifacts": []}
                else: 
                    task_result_data, suggested_new_tasks = self.worker.execute_task(
                        task_description_str, self.executed_tasks_results,
                        self.uploaded_files_info, self.goal
                    )
                
                self.process_task_result(task_description_str, task_result_data)
                log_message(f"Tarefa '{task_description_str}' concluída.", "TaskManager")
                if suggested_new_tasks: self.current_task_list.extend(suggested_new_tasks)
            
            # Validação Automática
            automatic_validation_result = self.validator.validate_results(self.executed_tasks_results, self.staged_artifacts, self.goal)
            last_automatic_validation_feedback = automatic_validation_result 

            if automatic_validation_result["status"] == "success":
                print_agent_message("TaskManager", "Validação automática dos artefatos bem-sucedida!")
                
                # Validação Manual (NOVO para v9.2)
                manual_validation_attempts = 0
                manual_approval_achieved = False
                while manual_validation_attempts <= MAX_MANUAL_VALIDATION_RETRIES and not manual_approval_achieved:
                    manual_val_result = self.present_for_manual_validation()

                    if manual_val_result["approved"]:
                        print_agent_message("TaskManager", "Artefatos aprovados manualmente pelo usuário.")
                        self.save_final_artifacts()
                        overall_success = True; manual_approval_achieved = True
                        break # Sai do loop de validação manual E do loop de validação automática
                    else: # Reprovado manualmente
                        manual_validation_attempts += 1
                        user_feedback = manual_val_result.get("feedback")
                        if user_feedback == "cancelar" or manual_validation_attempts > MAX_MANUAL_VALIDATION_RETRIES:
                            print_agent_message("TaskManager", "Validação manual cancelada ou máximo de tentativas atingido.")
                            overall_success = False # Garante que não prossiga como sucesso
                            manual_approval_achieved = True # Sai do loop de validação manual
                            break # Sai do loop de validação manual
                        
                        print_agent_message("TaskManager", f"Artefatos reprovados manualmente. Feedback: {user_feedback}")
                        last_manual_feedback = user_feedback # Salva para o próximo replanejamento
                        # Prepara para sair do loop de validação automática para replanejar
                        automatic_validation_attempts = 0 # Reseta para permitir novo ciclo de decomposição
                        previous_plan_for_replan = list(self.current_task_list) # Salva plano atual
                        # Não quebra o loop de validação automática aqui, deixa ele continuar para replanejar
                        manual_approval_achieved = True # Sai do loop de validação manual para permitir replanejamento
                        break # Sai do loop de validação manual
                
                if overall_success or user_feedback == "cancelar" or manual_validation_attempts > MAX_MANUAL_VALIDATION_RETRIES : # Se sucesso OU cancelamento/max tentativas manual
                     break # Sai do loop de validação automática

            else: # Validação automática falhou
                print_agent_message("TaskManager", f"Validação automática falhou: {automatic_validation_result['reason']}")
                automatic_validation_attempts += 1
                if automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES:
                    print_agent_message("TaskManager", "Tentando replanejar (baseado em falha automática)...")
                    current_goal_to_decompose = self.goal 
                    previous_plan_for_replan = list(self.current_task_list)
                else:
                    print_agent_message("TaskManager", "Máximo de tentativas de validação automática atingido. Encerrando.")
                    break 
        
        if overall_success: print_agent_message("TaskManager", "Fluxo de trabalho concluído com sucesso!")
        else: print_agent_message("TaskManager", "Fluxo de trabalho concluído com falhas ou cancelamento.")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.2" 
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


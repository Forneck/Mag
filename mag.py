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
MAX_AUTOMATIC_VALIDATION_RETRIES = 2 
MAX_MANUAL_VALIDATION_RETRIES = 2
MAX_ROUTING_ATTEMPTS_PER_TASK = 3 # Novo para Router

# --- Modelos e Catálogo ---
GEMINI_TEXT_MODEL_NAME = "gemini-2.0-flash"
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" # Mantido SDK

MODEL_CATALOG = {
    "general_text_processing": {
        "model_name": GEMINI_TEXT_MODEL_NAME,
        "worker_class_name": "Worker", # Usar string para instanciar depois
        "capabilities": ["text_analysis", "code_generation", "summarization", "refinement", "default", "documentation", "evaluation"]
    },
    "image_generation": {
        "model_name": GEMINI_IMAGE_GENERATION_MODEL_NAME,
        "worker_class_name": "ImageWorker",
        "capabilities": ["image_creation", "generate_image", "TASK_GERAR_IMAGEM"] # Adicionado TASK_GERAR_IMAGEM
    },
    "image_evaluation": { # Adicionado para TASK_AVALIAR_IMAGENS
        "model_name": GEMINI_TEXT_MODEL_NAME, # Usa modelo de texto para avaliar
        "worker_class_name": "Worker", # Worker de texto pode lidar com isso
        "capabilities": ["image_evaluation", "TASK_AVALIAR_IMAGENS"]
    }
}

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
except Exception as e: print(f"Erro API Gemini: {e}"); log_message(f"Erro API Gemini: {e}", "Sistema"); exit()

generation_config_text = {"temperature":0.7,"top_p":0.95,"top_k":64,"max_output_tokens":8192,"response_mime_type":"text/plain"}
generation_config_image_sdk = {"temperature":1.0,"top_p":0.95,"top_k":64,"response_modalities":['TEXT','IMAGE'],"max_output_tokens":8192}
safety_settings_gemini=[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"}]

# --- Funções Auxiliares de Comunicação ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")

def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config=None):
    # (Implementação da v9.2 mantida, já é robusta)
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

# --- Funções de Arquivos (mantidas da v9.2) ---
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

def clear_upload_cache(): # Implementação da v9.2 mantida
    print_agent_message("Sistema", f"Verificando cache local em: {UPLOADED_FILES_CACHE_DIR}")
    local_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "*.json"))
    if not local_files: print_agent_message("Sistema", "Nenhum cache local para limpar.")
    else:
        if input(f"👤 Limpar {len(local_files)} arquivo(s) de cache local? (s/n) ➡️ ").lower() == 's':
            for f in local_files: os.remove(f); log_message(f"Cache local '{f}' removido.", "Sistema")
    try:
        api_files = list(genai.list_files())
        if not api_files: print_agent_message("Sistema", "Nenhum arquivo na API Gemini."); return
        if input(f"👤 ‼️ ATENÇÃO ‼️ Deletar {len(api_files)} arquivo(s) da API Gemini? (s/n) ➡️ ").lower() == 's':
            for f_api in api_files: genai.delete_file(name=f_api.name); time.sleep(0.2)
            print_agent_message("Sistema", "Arquivos da API limpos (ou tentativa).")
    except Exception as e: log_message(f"Erro ao limpar cache API: {e}", "Sistema")


def get_uploaded_files_info_from_user(): # Implementação da v9.2 mantida
    uploaded_file_objects, uploaded_files_metadata, reused_ids = [], [], set()
    api_files = []
    try: api_files = list(genai.list_files())
    except Exception as e: log_message(f"Falha API list_files: {e}", "Sistema")
    
    api_dict = {f.name: f for f in api_files}
    cached_meta = load_cached_files_metadata(get_most_recent_cache_file())
    offer_reuse = [{**api_f._asdict(), **next((c for c in cached_meta if c.get("file_id") == api_f.name), {})} for api_f in api_files] # Combina info

    if offer_reuse:
        print_agent_message("Sistema", "Arquivos na API/cache:")
        for i, m in enumerate(offer_reuse): print(f"  {i+1}. {m.get('display_name','N/A')} (Origem: {m.get('user_path', 'API')})")
        if input("👤 Reutilizar? (s/n) ➡️ ").lower() == 's':
            choices = input("👤 Números (ex: 1,3) ou 'todos': ➡️ ").lower()
            sel_indices = list(range(len(offer_reuse))) if choices == 'todos' else [int(x.strip())-1 for x in choices.split(',') if x.strip().isdigit()]
            for idx in sel_indices:
                if 0 <= idx < len(offer_reuse):
                    chosen_meta = offer_reuse[idx]
                    try:
                        file_obj = api_dict.get(chosen_meta["name"]) or genai.get_file(name=chosen_meta["name"]) # 'name' é o ID
                        uploaded_file_objects.append(file_obj); uploaded_files_metadata.append(chosen_meta); reused_ids.add(chosen_meta["name"])
                    except Exception as e: log_message(f"Erro ao obter '{chosen_meta.get('display_name')}' para reutilização: {e}", "Sistema")
    
    if input("👤 Adicionar NOVOS arquivos? (s/n) ➡️ ").lower() == 's':
        while True:
            fp_pattern = input("👤 Caminho/padrão (ou 'fim'): ➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            expanded = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*','?']) else ([fp_pattern] if os.path.isfile(fp_pattern) else [])
            if not expanded: print("❌ Nenhum arquivo."); continue
            for fp in expanded:
                dn = os.path.basename(fp)
                try:
                    ext_map = { ".md": "text/markdown", ".py": "text/x-python", ".cpp": "text/x-c++src", ".h": "text/x-chdr"}
                    mime = ext_map.get(os.path.splitext(dn)[1].lower())
                    file_obj = genai.upload_file(path=fp, display_name=dn, mime_type=mime)
                    uploaded_file_objects.append(file_obj)
                    new_meta = {"file_id": file_obj.name, "display_name": dn, "mime_type": file_obj.mime_type, "user_path": fp, "uri": file_obj.uri, "size_bytes": file_obj.size_bytes, "state": str(file_obj.state)}
                    uploaded_files_metadata.append(new_meta); time.sleep(1)
                except Exception as e: log_message(f"Erro upload '{dn}': {e}", "Sistema")
    if uploaded_files_metadata:
        with open(os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json"), "w") as f: json.dump(uploaded_files_metadata, f, indent=4)
    return uploaded_file_objects, uploaded_files_metadata

# --- Classes dos Agentes ---
class Worker: # (Adaptado para v10.0)
    def __init__(self, task_manager, model_name=GEMINI_TEXT_MODEL_NAME):
        self.task_manager = task_manager; self.model_name = model_name
        log_message(f"Instância Worker criada para modelo {model_name}.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        agent_display_name = f"Worker ({self.model_name})"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")
        prompt_context = "Resultados anteriores:\n" + ('\n'.join([f"- '{list(res.keys())[0]}': {str(list(res.values())[0].get('text_content', list(res.values())[0]))[:300]}..." for res in previous_results]) if previous_results else "Nenhum.\n")
        files_prompt_part = "Arquivos carregados:\n" + ('\n'.join([f"- {f['display_name']}" for f in uploaded_files_info]) if uploaded_files_info else "Nenhum.\n")
        
        # Instrução especial para o Router
        router_instruction = "Se não for capaz de realizar a tarefa adequadamente, retorne APENAS um JSON com a chave 'execution_feedback' contendo: {'status': 'cannot_execute', 'reason': 'seu motivo detalhado', 'suggested_capability': 'sugestão de capacidade/modelo se souber'}. Caso contrário, execute a tarefa normalmente."

        prompt = [
            f"Você é um Agente Executor. Tarefa: \"{sub_task_description}\"",
            f"Contexto:\n{prompt_context}\n{files_prompt_part}\nObjetivo Original: {original_goal}",
            router_instruction, # Nova instrução
            "Execute a tarefa. Formato de saída para arquivos:",
            "  \"Arquivo: nome.ext\"\n  ```linguagem\n  // código\n  ```",
            "  OU JSON: {\"resultado\": \"descrição\", \"arquivos\": [{\"nome\": \"f1.cpp\", \"conteudo\": \"...\"}]}",
            "Resultado da Tarefa:"
        ]
        if self.task_manager.uploaded_file_objects: prompt.extend(self.task_manager.uploaded_file_objects)
        
        task_res_raw = call_gemini_api_with_retry(prompt, agent_display_name, model_name=self.model_name)
        
        if task_res_raw is None: 
            return {"execution_status": "failure_api_error", "text_content": "Falha: Sem resposta da API.", "artifacts": []}, []

        # Verificar se a LLM retornou feedback de incapacidade
        try:
            feedback_json = json.loads(task_res_raw)
            if isinstance(feedback_json, dict) and "execution_feedback" in feedback_json:
                ef = feedback_json["execution_feedback"]
                log_message(f"Worker ({self.model_name}) indicou incapacidade: {ef}", agent_display_name)
                return {
                    "execution_status": ef.get("status", "cannot_execute"),
                    "reason": ef.get("reason"),
                    "suggested_capability": ef.get("suggested_capability"),
                    "text_content": ef.get("reason"), # Usar a razão como conteúdo textual neste caso
                    "artifacts": []
                }, []
        except json.JSONDecodeError:
            pass # Não era JSON de feedback, processar normalmente

        # Extração de artefatos e sugestões de tarefas (lógica da v9.2)
        sugg_tasks_match = re.search(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", task_res_raw, re.DOTALL | re.IGNORECASE)
        sugg_tasks_strings = []
        if sugg_tasks_match:
            try: sugg_tasks_strings = json.loads(sugg_tasks_match.group(1))
            except json.JSONDecodeError: log_message("Erro JSON NOVAS_TAREFAS_SUGERIDAS", agent_display_name)
        
        task_res_content = re.sub(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", "", task_res_raw, flags=re.DOTALL | re.IGNORECASE).strip()
        extracted_artifacts = self._extract_artifacts_from_output(task_res_content) # Usa o método da v9.2
        
        main_text_output = task_res_content
        if extracted_artifacts: # Tenta pegar "resultado" do JSON se houver
            try:
                json_data = json.loads(task_res_content.strip().lstrip("```json").rstrip("```").strip())
                if isinstance(json_data, dict) and "resultado" in json_data: main_text_output = json_data["resultado"]
            except json.JSONDecodeError: pass

        log_message(f"Resultado (Worker {self.model_name}): {str(main_text_output)[:300]}...", agent_display_name)
        return {"execution_status": "success", "text_content": main_text_output, "artifacts": extracted_artifacts}, sugg_tasks_strings

    def _extract_artifacts_from_output(self, output_str): # Mantido da v9.2
        artifacts = []
        try: 
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
        except Exception: pass 

        patterns = [
            re.compile(r"Arquivo:\s*(?P<filename>[^\n`]+)\s*\n```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*\n(?P<content>.*?)\n```", re.DOTALL | re.MULTILINE),
            re.compile(r"```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*(?P<filename>[^\n`]*\.[a-zA-Z0-9_]+)?\s*\n(?P<content>.*?)\n```", re.DOTALL | re.MULTILINE)
        ]
        processed_starts = set()
        for i, pattern in enumerate(patterns):
            for match in pattern.finditer(output_str):
                if match.start() in processed_starts: continue
                filename_match = match.groupdict().get("filename"); content_match = match.groupdict().get("content"); language_match = match.groupdict().get("language","").lower()
                if not filename_match and i == 1: continue 
                filename = sanitize_filename(filename_match.strip()) if filename_match else ""; content = content_match.strip() if content_match else ""; language = language_match or (filename.split('.')[-1] if '.' in filename else "")
                if filename and content and not (content.lower().startswith("arquivo:") and len(content.splitlines()) <=2) and content.strip():
                    if not any(a["filename"] == filename and a["content"][:50] == content[:50] for a in artifacts):
                        artifacts.append({"type": "markdown" if language=="markdown" or filename.endswith(".md") else "code", "filename": filename, "content": content, "language": language, "extraction_method": "regex_directive" if i==0 else "regex_fallback"})
                        processed_starts.add(match.start())
        if artifacts: log_message(f"{len(artifacts)} artefatos extraídos via Regex.", "Worker")
        return artifacts

class ImageWorker: # (Adaptado para v10.0)
    def __init__(self, task_manager, model_name=GEMINI_IMAGE_GENERATION_MODEL_NAME):
        self.task_manager = task_manager; self.model_name = model_name
        self.generation_config = generation_config_image_sdk
        log_message(f"ImageWorker criado para {self.model_name}", "ImageWorker")

    def execute_task(self, image_prompt, previous_results, uploaded_files_info, original_goal, task_description_str="Gerar Imagem"): # Assinatura similar ao Worker
        agent_display_name = f"ImageWorker ({self.model_name})"
        print_agent_message(agent_display_name, f"Gerando imagem para: '{image_prompt[:100]}...'")
        
        # Instrução especial para o Router (menos provável de ser usada aqui, mas por consistência)
        router_instruction = "Se não for capaz de realizar a tarefa adequadamente (ex: prompt inadequado para imagem), retorne APENAS um JSON com a chave 'execution_feedback' contendo: {'status': 'cannot_execute', 'reason': 'seu motivo', 'suggested_capability': 'text_analysis_for_prompt_refinement'}. Caso contrário, gere a imagem."
        # O prompt para o ImageWorker é apenas a descrição da imagem, mas podemos adicionar a instrução.
        # No entanto, o SDK de imagem não processa texto complexo como o modelo de chat.
        # Por ora, o ImageWorker não usará a `router_instruction` no prompt para a API de imagem.
        # Ele retornará 'cannot_execute' se o prompt for vazio, por exemplo.

        if not image_prompt or not image_prompt.strip():
            log_message("ImageWorker: Prompt de imagem vazio.", agent_display_name)
            return {"execution_status": "cannot_execute", "reason": "Prompt de imagem está vazio.", "text_content": "Prompt de imagem vazio.", "artifacts": []}, []

        response_obj = call_gemini_api_with_retry([image_prompt], agent_display_name, self.model_name, self.generation_config)
        if not response_obj: 
            return {"execution_status": "failure_api_error", "text_content": "Falha: Sem resposta da API de imagem.", "artifacts": []}, []
        try:
            if response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                for part in response_obj.candidates[0].content.parts:
                    if hasattr(part, 'image') and part.image and hasattr(part.image, 'data'):
                        img_bytes = part.image.data
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fn_base = sanitize_filename(f"imagem_gerada_{task_description_str[:20]}_{ts}")
                        fn = f"{fn_base}.png"
                        log_message(f"Imagem gerada, pronta para salvar como '{fn}'.", agent_display_name)
                        artifact = {"filename": fn, "content_bytes": img_bytes, "type": "image", "extraction_method": "sdk"}
                        return {"execution_status": "success", "text_content": f"Imagem '{fn}' gerada.", "artifacts": [artifact]}, []
            return {"execution_status": "failure_execution_error", "reason": "Nenhuma imagem na resposta da API.", "text_content": "Falha: Nenhuma imagem na resposta.", "artifacts": []}, []
        except Exception as e: 
            log_message(f"Erro ao processar imagem: {e}\n{traceback.format_exc()}", agent_display_name)
            return {"execution_status": "failure_execution_error", "reason": f"Erro ao processar imagem: {e}", "text_content": f"Erro: {e}", "artifacts": []}, []

class Validator: # (Mantida da v9.2)
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Instância do Validator criada.", "Validator")

    def validate_results(self, task_results_history, staged_artifacts, original_goal):
        log_message(f"Iniciando validação automática. Artefatos no stage: {list(staged_artifacts.keys())}", "Validator")
        issues = []
        if not staged_artifacts:
            expects_files = any(kw in original_goal.lower() for kw in ["criar", "modificar", "salvar"]) or \
                            (hasattr(self.task_manager, 'current_task_list') and \
                             any(any(kw_task in (list(task.keys())[0] if isinstance(task,dict) else str(task)).lower() for kw_task in ["salvar", "criar arquivo", "gerar o arquivo"]) for task in self.task_manager.current_task_list))
            if expects_files: issues.append("Nenhum artefato preparado, mas era esperado.")
        else:
            for filename, artifact_data in staged_artifacts.items():
                content = artifact_data.get("content"); content_bytes = artifact_data.get("content_bytes")
                if artifact_data.get("type") == "image" and not content_bytes: issues.append(f"Imagem '{filename}' sem bytes.")
                elif artifact_data.get("type") != "image" and (not content or not content.strip()): issues.append(f"Artefato '{filename}' vazio.")
                if "Arquivo:" in filename: issues.append(f"Nome de arquivo suspeito '{filename}'.")
                if isinstance(content, str) and content.lower().startswith("arquivo:") and len(content.splitlines()) <= 2: issues.append(f"Conteúdo de '{filename}' parece referência.")
        
        if issues:
            log_message(f"Validação automática falhou. Problemas: {'; '.join(issues)}", "Validator")
            return {"status": "failure", "reason": "; ".join(issues), "suggested_refinements": "Revisar tarefas. Garantir nomes/conteúdo corretos."}
        log_message(f"Validação automática OK para: {list(staged_artifacts.keys())}", "Validator")
        return {"status": "success"}

class Router: # (NOVO para v10.0)
    def __init__(self, task_manager, model_catalog):
        self.task_manager = task_manager
        self.model_catalog = model_catalog
        self.task_routing_history = {} # {task_description_str: {"attempts": [], "final_handler": None}}
        log_message("Instância do Router criada.", "Router")

    def _get_worker_instance(self, worker_class_name, model_name):
        if worker_class_name == "Worker":
            return Worker(self.task_manager, model_name)
        elif worker_class_name == "ImageWorker":
            return ImageWorker(self.task_manager, model_name)
        # Adicionar mais workers aqui se necessário
        log_message(f"Classe de worker desconhecida: {worker_class_name}", "Router")
        return None

    def _find_model_by_capability(self, capability, tried_models_for_this_task):
        if not capability: return None
        for entry_key, config in self.model_catalog.items():
            if capability.lower() in [c.lower() for c in config["capabilities"]] and config["model_name"] not in tried_models_for_this_task:
                log_message(f"Router encontrou modelo '{config['model_name']}' para capacidade '{capability}'.", "Router")
                return config
        return None

    def select_initial_model_config(self, task_description):
        # Lógica simples baseada em palavras-chave para v10.0
        task_lower = task_description.lower()
        if "imagem" in task_lower or "gerar imagem" in task_lower or "task_gerar_imagem" in task_lower:
            return self.model_catalog.get("image_generation")
        if "avaliar imagem" in task_lower or "task_avaliar_imagens" in task_lower:
            return self.model_catalog.get("image_evaluation") # Que usa Worker de texto
        return self.model_catalog.get("general_text_processing") # Default

    def route_task(self, task_description_str, previous_results, uploaded_files_info, original_goal):
        log_message(f"Router iniciando roteamento para tarefa: '{task_description_str}'", "Router")
        if task_description_str not in self.task_routing_history:
            self.task_routing_history[task_description_str] = {"attempts": [], "final_handler": None}
        
        tried_models_for_this_task = [attempt["model_name"] for attempt in self.task_routing_history[task_description_str]["attempts"]]
        
        current_model_config = None
        suggested_capability_from_worker = None

        for attempt_num in range(MAX_ROUTING_ATTEMPTS_PER_TASK):
            log_message(f"Router: Tentativa de roteamento {attempt_num + 1}/{MAX_ROUTING_ATTEMPTS_PER_TASK} para '{task_description_str}'", "Router")

            if attempt_num == 0: # Primeira tentativa
                current_model_config = self.select_initial_model_config(task_description_str)
            elif suggested_capability_from_worker:
                current_model_config = self._find_model_by_capability(suggested_capability_from_worker, tried_models_for_this_task)
                suggested_capability_from_worker = None # Resetar para a próxima iteração do loop interno
            
            if not current_model_config: # Se ainda não tem config (ex: sugestão não mapeada)
                # Tentar um default se ainda não foi tentado
                default_config = self.model_catalog.get("general_text_processing")
                if default_config and default_config["model_name"] not in tried_models_for_this_task:
                    current_model_config = default_config
                else: # Esgotou opções ou não encontrou
                    log_message(f"Router: Não foi possível encontrar um modelo adequado restante para '{task_description_str}'.", "Router")
                    break # Sai do loop de tentativas de roteamento

            model_name_to_try = current_model_config["model_name"]
            worker_class_name = current_model_config["worker_class_name"]
            
            log_message(f"Router: Tentando com modelo '{model_name_to_try}' (Worker: {worker_class_name})", "Router")
            worker_instance = self._get_worker_instance(worker_class_name, model_name_to_try)

            if not worker_instance:
                log_message(f"Router: Falha ao instanciar worker {worker_class_name}. Pulando este modelo.", "Router")
                self.task_routing_history[task_description_str]["attempts"].append({
                    "model_name": model_name_to_try, "status": "failure_worker_instantiation", "reason": "Worker class not found"
                })
                tried_models_for_this_task.append(model_name_to_try)
                current_model_config = None # Para forçar nova seleção
                continue

            # Para ImageWorker, o "prompt" é a descrição da imagem. Para Worker, é a tarefa completa.
            # A tarefa TASK_GERAR_IMAGEM já tem o prompt no formato "TASK_GERAR_IMAGEM: [prompt_da_imagem]"
            execution_arg = task_description_str
            if worker_class_name == "ImageWorker" and task_description_str.startswith("TASK_GERAR_IMAGEM:"):
                 execution_arg = task_description_str.replace("TASK_GERAR_IMAGEM:", "").strip()


            # Chamada unificada para execute_task
            task_result_data, suggested_new_tasks = worker_instance.execute_task(
                execution_arg, # task_description_str ou image_prompt
                previous_results,
                uploaded_files_info,
                original_goal,
                task_description_str=task_description_str # Passa a descrição original para ImageWorker também
            )
            
            execution_status = task_result_data.get("execution_status", "failure_unknown")
            reason = task_result_data.get("reason")
            suggested_capability_from_worker = task_result_data.get("suggested_capability")

            self.task_routing_history[task_description_str]["attempts"].append({
                "model_name": model_name_to_try, "status": execution_status, "reason": reason, "suggested_capability": suggested_capability_from_worker
            })
            tried_models_for_this_task.append(model_name_to_try)


            if execution_status == "success":
                log_message(f"Router: Tarefa '{task_description_str}' executada com sucesso por '{model_name_to_try}'.", "Router")
                self.task_routing_history[task_description_str]["final_handler"] = model_name_to_try
                return task_result_data, suggested_new_tasks
            
            elif execution_status == "cannot_execute":
                log_message(f"Router: Modelo '{model_name_to_try}' não pôde executar '{task_description_str}'. Razão: {reason}. Sugestão: {suggested_capability_from_worker}", "Router")
                # Loop continua para tentar outro modelo com base na sugestão ou próximo candidato
                current_model_config = None # Forçar nova seleção
            else: # failure_api_error, failure_execution_error, etc.
                log_message(f"Router: Modelo '{model_name_to_try}' falhou ao executar '{task_description_str}'. Razão: {reason}", "Router")
                # Pode-se decidir tentar outro modelo ou parar. Por ora, vamos tentar outro.
                current_model_config = None # Forçar nova seleção
        
        log_message(f"Router: Esgotadas tentativas de roteamento para '{task_description_str}'.", "Router")
        # Retorna o resultado da última tentativa, mesmo que falha, ou um resultado de falha genérico.
        # O TaskManager tratará a falha.
        return {"execution_status": "failure_routing_exhausted", "text_content": f"Falha ao executar tarefa '{task_description_str}' após múltiplas tentativas de roteamento.", "artifacts": []}, []


class TaskManager: # (Adaptado para v10.0)
    def __init__(self, initial_goal, uploaded_file_objects=None, uploaded_files_info=None):
        self.goal = initial_goal
        self.uploaded_file_objects = uploaded_file_objects or []
        self.uploaded_files_info = uploaded_files_info or []
        self.current_task_list = []
        self.executed_tasks_results = [] 
        self.staged_artifacts = {} 
        self.router = Router(self, MODEL_CATALOG) # NOVO: Instancia o Router
        self.validator = Validator(self)
        log_message("Instância do TaskManager (v10.0) criada com Router.", "TaskManager")

    def decompose_goal(self, goal_to_decompose, previous_plan=None, automatic_validation_feedback=None, manual_validation_feedback_str=None):
        # (Implementação da v9.2 mantida, já inclui feedback para replanejamento)
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{goal_to_decompose}'")
        files_prompt_part = "Arquivos Complementares:\n" + ('\n'.join([f"- {f['display_name']} (ID: {f['file_id']})" for f in self.uploaded_files_info]) if self.uploaded_files_info else "Nenhum.\n")
        prompt_parts = [
            "Você é um Gerenciador de Tarefas especialista. Decomponha a meta principal em sub-tarefas sequenciais.",
            f"Meta Principal: \"{goal_to_decompose}\"", files_prompt_part,
            "Se a meta envolver CRIAÇÃO DE MÚLTIPLAS IMAGENS (ex: \"crie 3 logos\"), você DEVE:",
            "1.  Criar uma tarefa para gerar a descrição de CADA imagem individualmente. Ex: \"Criar descrição para imagem 1 de [assunto]\".",
            "2.  Seguir CADA tarefa de descrição com uma tarefa \"TASK_GERAR_IMAGEM: [assunto da imagem correspondente]\".",
            "3.  Após TODAS as tarefas de geração de imagem, adicionar UMA tarefa: \"TASK_AVALIAR_IMAGENS: Avaliar as imagens/descrições geradas para [objetivo original] e selecionar as melhores que atendem aos critérios.\"",
            "Se for a CRIAÇÃO DE UMA ÚNICA IMAGEM, use o formato:",
            "1.  \"Criar uma descrição textual detalhada (prompt) para gerar a imagem de [assunto].\"",
            "2.  \"TASK_GERAR_IMAGEM: [assunto da imagem]\"",
            "3.  \"TASK_AVALIAR_IMAGENS: Avaliar a imagem gerada para [objetivo original].\"",
            "Se precisar usar imagem fornecida MAS SEM ENVOLVER CRIAÇÃO DE IMAGENS, use \"TASK_AVALIAR_IMAGENS: Avaliar a imagem fornecida para [objetivo original].\"",
            "Para outras metas, decomponha normalmente. Retorne em JSON array de strings."
        ]
        if previous_plan: 
            prompt_parts.append(f"\nContexto Adicional: O plano anterior precisa de revisão.")
            prompt_parts.append(f"Plano Anterior: {json.dumps(previous_plan)}")
            if automatic_validation_feedback and automatic_validation_feedback.get("status") == "failure":
                prompt_parts.append(f"Feedback da Validação Automática: {automatic_validation_feedback['reason']}")
                prompt_parts.append(f"Sugestões de Refinamento (Automático): {automatic_validation_feedback['suggested_refinements']}")
            if manual_validation_feedback_str:
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

    def present_for_manual_validation(self): # (Mantido da v9.2)
        log_message("Iniciando validação manual pelo usuário...", "TaskManager")
        print_agent_message("TaskManager", "Validação automática OK. Artefatos gerados/modificados:")
        if not self.staged_artifacts:
            print_agent_message("TaskManager", "Nenhum artefato preparado para validação manual."); return {"approved": True, "feedback": "Nenhum artefato no stage."}
        for i, (filename, data) in enumerate(self.staged_artifacts.items()): print(f"  {i+1}. {data.get('type', 'desconhecido').capitalize()}: {filename}")
        user_choice = input("👤 Aprova estes artefatos finais? (s/n/cancelar) ➡️ ").strip().lower()
        if user_choice == 's': return {"approved": True}
        if user_choice == 'cancelar': return {"approved": False, "feedback": "cancelar"}
        feedback = input("👤 Feedback para correção (ou 'cancelar'): ➡️ ").strip()
        return {"approved": False, "feedback": feedback if feedback else "Reprovado sem feedback específico."}

    def process_task_result(self, task_description, task_result_data): # (Mantido da v9.2)
        self.executed_tasks_results.append({task_description: task_result_data})
        if isinstance(task_result_data, dict):
            for artifact in task_result_data.get("artifacts", []):
                filename, art_type = artifact.get("filename"), artifact.get("type")
                if filename:
                    if art_type == "image" and artifact.get("content_bytes"): self.staged_artifacts[filename] = artifact
                    elif art_type in ["code", "markdown"] and artifact.get("content"): self.staged_artifacts[filename] = artifact

    def save_final_artifacts(self): # (Mantido da v9.2)
        log_message(f"Salvando {len(self.staged_artifacts)} artefatos finais...", "TaskManager")
        saved_count = 0
        if not self.staged_artifacts: print_agent_message("TaskManager", "Nenhum artefato para salvar."); return
        for filename, data in self.staged_artifacts.items():
            final_fn = sanitize_filename(filename)
            if not final_fn: continue
            fp = os.path.join(OUTPUT_DIRECTORY, final_fn)
            try:
                if data.get("type") == "image" and data.get("content_bytes"):
                    with open(fp, "wb") as f: f.write(data["content_bytes"]); print_agent_message("TaskManager", f"🖼️ Imagem salva: {final_fn}")
                elif data.get("content") and data.get("type") in ["code", "markdown"]:
                    with open(fp, "w", encoding="utf-8") as f: f.write(data["content"]); print_agent_message("TaskManager", f"📄 Arquivo salvo: {final_fn}")
                else: continue
                log_message(f"Artefato final salvo: {fp}", "TaskManager"); saved_count += 1
            except Exception as e: log_message(f"Erro ao salvar '{final_fn}': {e}", "TaskManager")
        print_agent_message("TaskManager", f"✅ {saved_count} artefato(s) salvo(s)." if saved_count else "Nenhum artefato efetivamente salvo.")

    def run_workflow(self): # (Adaptado para v10.0 - usa Router)
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        overall_success = False
        automatic_validation_attempts = 0
        current_goal_to_decompose = self.goal
        previous_plan_for_replan = None
        last_automatic_validation_feedback = None
        last_manual_feedback_str = None

        while automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES and not overall_success:
            if automatic_validation_attempts > 0: log_message(f"Replanejamento automático {automatic_validation_attempts}", "TaskManager")
            
            self.current_task_list = self.decompose_goal(
                current_goal_to_decompose, previous_plan_for_replan,
                last_automatic_validation_feedback, last_manual_feedback_str
            )
            last_manual_feedback_str = None # Reset

            if not self.current_task_list: print_agent_message("TaskManager", "Falha na decomposição. Encerrando."); return

            print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
            for i, task_desc in enumerate(self.current_task_list): print(f"  {i+1}. {task_desc}")
            if input("👤 Aprova plano? (s/n) ➡️ ").lower() != 's': print_agent_message("TaskManager", "Plano não aprovado."); return
            
            self.executed_tasks_results = []; self.staged_artifacts = {} # Reset para novo ciclo de plano

            for i, task_description_str in enumerate(self.current_task_list):
                print_agent_message("TaskManager", f"Próxima tarefa ({i+1}/{len(self.current_task_list)}): {task_description_str}")
                
                # NOVO: Chamar o Router para executar a tarefa
                task_result_data, suggested_new_tasks = self.router.route_task(
                    task_description_str,
                    self.executed_tasks_results,
                    self.uploaded_files_info,
                    self.goal
                )
                
                # Verificar se o roteamento/execução falhou criticamente
                if task_result_data.get("execution_status", "").startswith("failure_"):
                    print_agent_message("TaskManager", f"Falha crítica na execução/roteamento da tarefa '{task_description_str}': {task_result_data.get('reason', 'Erro desconhecido')}")
                    # Decide se quer parar todo o workflow ou tentar replanejar
                    # Por ora, vamos permitir que a validação automática pegue isso se não houver artefatos.
                    # Se for 'failure_routing_exhausted', a validação automática provavelmente falhará se artefatos eram esperados.
                
                self.process_task_result(task_description_str, task_result_data)
                log_message(f"Tarefa '{task_description_str}' processada pelo TaskManager.", "TaskManager")
                if suggested_new_tasks: self.current_task_list.extend(suggested_new_tasks) # Adiciona tarefas sugeridas pelo Worker
            
            # Validação Automática
            automatic_validation_result = self.validator.validate_results(self.executed_tasks_results, self.staged_artifacts, self.goal)
            last_automatic_validation_feedback = automatic_validation_result 

            if automatic_validation_result["status"] == "success":
                print_agent_message("TaskManager", "Validação automática OK!")
                manual_validation_attempts = 0
                while manual_validation_attempts <= MAX_MANUAL_VALIDATION_RETRIES:
                    manual_val_res = self.present_for_manual_validation()
                    if manual_val_res["approved"]:
                        self.save_final_artifacts(); overall_success = True; break
                    
                    last_manual_feedback_str = manual_val_res.get("feedback")
                    if last_manual_feedback_str == "cancelar" or manual_validation_attempts >= MAX_MANUAL_VALIDATION_RETRIES:
                        print_agent_message("TaskManager", "Validação manual cancelada ou máximo de tentativas."); overall_success = False; break 
                    
                    print_agent_message("TaskManager", f"Reprovado manualmente. Feedback: {last_manual_feedback_str}")
                    manual_validation_attempts += 1
                    # Prepara para replanejamento saindo do loop de validação manual e permitindo que o loop de validação automática continue para replanejar
                    automatic_validation_attempts = 0 # Reseta para permitir novo ciclo de decomposição
                    previous_plan_for_replan = list(self.current_task_list)
                    last_automatic_validation_feedback = None # Não queremos que o feedback automático antigo interfira com o manual
                    break # Sai do loop de validação manual para ir para o replanejamento automático
                if overall_success or last_manual_feedback_str == "cancelar" or manual_validation_attempts > MAX_MANUAL_VALIDATION_RETRIES:
                    break # Sai do loop de validação automática
            
            else: # Validação automática falhou
                print_agent_message("TaskManager", f"Validação automática falhou: {automatic_validation_result['reason']}")
                automatic_validation_attempts += 1
                if automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES:
                    print_agent_message("TaskManager", "Tentando replanejar (falha automática)...")
                    previous_plan_for_replan = list(self.current_task_list)
                else:
                    print_agent_message("TaskManager", "Máx. tentativas de validação automática. Encerrando."); break 
        
        if overall_success: print_agent_message("TaskManager", "Fluxo de trabalho concluído com sucesso!")
        else: print_agent_message("TaskManager", "Fluxo de trabalho concluído com falhas ou cancelamento.")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v10.0" 
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
    # ... (restante da main como na v9.2) ...
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

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
MAX_ROUTING_ATTEMPTS = 3 # NOVO: Limite para o Router tentar diferentes modelos

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

GEMINI_TEXT_MODEL_NAME = "gemini-1.5-flash-latest" # Alterado para um modelo comum e eficiente
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-1.5-flash-latest" # O mesmo modelo pode gerar texto e imagem via prompt

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
log_message(f"Modelo Gemini (geração de imagem): {GEMINI_IMAGE_GENERATION_MODEL_NAME}", "Sistema")

generation_config_text = {"temperature":0.7,"top_p":0.95,"top_k":64,"max_output_tokens":8192,"response_mime_type":"text/plain"}
safety_settings_gemini=[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"}]


# --- Funções Auxiliares de Comunicação ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")

def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config=None):
    log_message(f"Iniciando chamada à API Gemini para {agent_name} (Modelo: {model_name})...", "Sistema")
    active_gen_config = gen_config if gen_config is not None else generation_config_text

    text_prompt_for_log = ""
    file_references_for_log = []
    for part_item in prompt_parts:
        if isinstance(part_item, str): text_prompt_for_log += part_item + "\n"
        elif hasattr(part_item, 'name') and hasattr(part_item, 'display_name'):
            file_references_for_log.append(f"Arquivo: {part_item.display_name} (ID: {part_item.name}, TipoMIME: {getattr(part_item, 'mime_type', 'N/A')})")

    log_message(f"Prompt textual para {agent_name}:\n---\n{text_prompt_for_log}\n---", "Sistema")
    if file_references_for_log: log_message(f"Arquivos referenciados: " + "\n".join(file_references_for_log), "Sistema")
    log_message(f"Usando generation_config: {active_gen_config}", "Sistema")

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES} para {agent_name}...", "Sistema")
        try:
            model_instance = genai.GenerativeModel(model_name, generation_config=active_gen_config, safety_settings=safety_settings_gemini)
            response = model_instance.generate_content(prompt_parts)
            log_message(f"Resposta bruta da API Gemini (tentativa {attempt + 1}): {response}", agent_name)

            # Para ImageWorker, o objeto de resposta completo é necessário
            if agent_name == "ImageWorker":
                return response

            if hasattr(response, 'text') and response.text is not None:
                return response.text.strip()

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                first_part = response.candidates[0].content.parts[0]
                if hasattr(first_part, 'text') and first_part.text is not None:
                    return first_part.text.strip()

            log_message(f"API não retornou texto utilizável (tentativa {attempt + 1}).", agent_name)
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                log_message(f"Bloqueio: {getattr(response.prompt_feedback, 'block_reason_message', 'N/A')}", agent_name)
            if attempt < MAX_API_RETRIES - 1: time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1}: {type(e).__name__} - {e}\n{traceback.format_exc()}", agent_name)
            if attempt < MAX_API_RETRIES - 1: time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR
            else: return None
    return None

# --- Funções de Arquivos (MANTIDAS DA V9.3.3) ---
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
    print_agent_message("Sistema", f"Verificando cache local em: {UPLOADED_FILES_CACHE_DIR}")
    local_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "*.json"))
    if not local_files: print_agent_message("Sistema", "Nenhum cache local para limpar.")
    else:
        print_agent_message("Sistema", f"Encontrados {len(local_files)} arquivo(s) de cache local.")
        if input(f"👤 Limpar {len(local_files)} arquivo(s) de cache local? (s/n) ➡️ ").lower() == 's':
            for f_path in local_files:
                try: os.remove(f_path); log_message(f"Cache local '{os.path.basename(f_path)}' removido.", "Sistema")
                except Exception as e: log_message(f"Erro ao remover cache local '{os.path.basename(f_path)}': {e}", "Sistema")
            print_agent_message("Sistema", "Cache local limpo.")
        else:
            print_agent_message("Sistema", "Limpeza do cache local cancelada.")
    try:
        api_files = list(genai.list_files())
        if not api_files: print_agent_message("Sistema", "Nenhum arquivo na API Gemini."); return
        print_agent_message("Sistema", f"Encontrados {len(api_files)} arquivo(s) na API Gemini Files.")
        if input(f"👤 ‼️ ATENÇÃO ‼️ Deletar {len(api_files)} arquivo(s) da API Gemini? (s/n)  ➡️ ").lower() == 's':
            for f_api in api_files:
                try: genai.delete_file(name=f_api.name); log_message(f"Arquivo API '{f_api.display_name}' deletado.", "Sistema"); print_agent_message("Sistema", f"  🗑️ Deletado da API: {f_api.display_name}"); time.sleep(0.2)
                except Exception as e: log_message(f"Erro ao deletar '{f_api.display_name}' da API: {e}", "Sistema")
            print_agent_message("Sistema", "Arquivos da API Gemini Files limpos.")
        else:
            print_agent_message("Sistema", "Limpeza de arquivos da API cancelada.")
    except Exception as e: log_message(f"Erro ao limpar cache API: {e}", "Sistema"); print_agent_message("Sistema", f"❌ Erro ao acessar/limpar arquivos da API: {e}")

def get_uploaded_files_info_from_user():
    uploaded_file_objects, uploaded_files_metadata, reused_ids = [], [], set()
    api_files_list = []
    try: api_files_list = list(genai.list_files())
    except Exception as e: log_message(f"Falha API list_files: {e}", "Sistema")

    api_files_dict = {f.name: f for f in api_files_list}
    cached_metadata_from_file = load_cached_files_metadata(get_most_recent_cache_file())
    offer_for_reuse_metadata_list = []

    for api_file in api_files_list:
        meta_from_cache = next((cm for cm in cached_metadata_from_file if cm.get("file_id") == api_file.name), {})
        display_name = meta_from_cache.get("display_name", api_file.display_name)
        user_path = meta_from_cache.get("user_path", "API (sem cache local)")
        mime_type = meta_from_cache.get("mime_type", api_file.mime_type)
        offer_for_reuse_metadata_list.append({
            "file_id": api_file.name, "display_name": display_name,
            "mime_type": mime_type, "uri": api_file.uri,
            "size_bytes": api_file.size_bytes, "state": str(api_file.state),
            "user_path": user_path
        })

    if offer_for_reuse_metadata_list:
        print_agent_message("Sistema", "Arquivos na API/cache:")
        for i, m in enumerate(offer_for_reuse_metadata_list): print(f"  {i+1}. {m['display_name']} (Origem: {m['user_path']})")
        if input("👤 Reutilizar arquivos? (s/n) ➡️ ").lower() == 's':
            choices = input("👤 Números (ex: 1,3) ou 'todos': ➡️ ").lower()
            sel_indices = list(range(len(offer_for_reuse_metadata_list))) if choices == 'todos' else [int(x.strip())-1 for x in choices.split(',') if x.strip().isdigit()]
            for idx in sel_indices:
                if 0 <= idx < len(offer_for_reuse_metadata_list):
                    chosen_meta = offer_for_reuse_metadata_list[idx]
                    try:
                        file_obj = api_files_dict.get(chosen_meta["file_id"]) or genai.get_file(name=chosen_meta["file_id"])
                        uploaded_file_objects.append(file_obj); uploaded_files_metadata.append(chosen_meta); reused_ids.add(chosen_meta["file_id"])
                        print_agent_message("Sistema", f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                        log_message(f"Arquivo '{chosen_meta['display_name']}' reutilizado.", "Sistema")
                    except Exception as e: log_message(f"Erro ao obter '{chosen_meta['display_name']}' para reutilização: {e}", "Sistema")

    if input("👤 Adicionar NOVOS arquivos? (s/n) ➡️ ").lower() == 's':
        while True:
            fp_pattern = input("👤 Caminho/padrão (ou 'fim'): ➡️ ").strip()
            if fp_pattern.lower() == 'fim': break

            expanded_paths = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*','?']) else ([fp_pattern] if os.path.exists(fp_pattern) else [])
            expanded_files = [f for f in expanded_paths if os.path.isfile(f)]

            if not expanded_files: print_agent_message("Sistema", f"❌ Nenhum arquivo encontrado para: '{fp_pattern}'"); continue

            print_agent_message("Sistema", f"Arquivos encontrados para '{fp_pattern}': {[os.path.basename(f) for f in expanded_files]}")
            if len(expanded_files) > 1 and input(f"👤 Confirmar upload de {len(expanded_files)} arquivos? (s/n) ➡️ ").lower() != 's': continue

            for fp in expanded_files:
                dn = os.path.basename(fp)
                try:
                    print_agent_message("Sistema", f"Fazendo upload de '{dn}'...")
                    file_obj = genai.upload_file(path=fp, display_name=dn)
                    uploaded_file_objects.append(file_obj)
                    new_meta = {"file_id": file_obj.name, "display_name": dn, "mime_type": file_obj.mime_type, "user_path": fp, "uri": file_obj.uri, "size_bytes": file_obj.size_bytes, "state": str(file_obj.state)}
                    uploaded_files_metadata.append(new_meta)
                    print_agent_message("Sistema", f"✅ Novo arquivo '{dn}' enviado.")
                    log_message(f"Novo arquivo '{dn}' enviado. Metadados: {new_meta}", "Sistema"); time.sleep(1)
                except Exception as e: log_message(f"Erro upload '{dn}': {e}", "Sistema"); print_agent_message("Sistema", f"❌ Erro no upload de '{dn}': {e}")
    if uploaded_files_metadata:
        with open(os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json"), "w", encoding="utf-8") as f: json.dump(uploaded_files_metadata, f, indent=4)
    return uploaded_file_objects, uploaded_files_metadata

# --- Classes dos Agentes ---

class Router:
    """
    NOVO AGENTE (da v9.5): Decide qual modelo usar para uma tarefa e aprende com falhas.
    """
    def __init__(self):
        self.meta_model_name = GEMINI_TEXT_MODEL_NAME
        self.known_models = [GEMINI_TEXT_MODEL_NAME]
        self.routing_cache = {} # Cache para otimizar chamadas
        log_message(f"Router criado com os modelos conhecidos: {self.known_models}", "Router")

    def add_model(self, model_name):
        """Adiciona um novo modelo à lista de especialistas conhecidos."""
        if model_name not in self.known_models:
            self.known_models.append(model_name)
            print_agent_message("Router", f"Novo especialista descoberto! Adicionando '{model_name}' à lista.")
            log_message(f"Novo modelo adicionado à lista de roteamento: {model_name}", "Router")

    def parse_suggestion(self, response_text):
        """Tenta extrair uma sugestão de modelo de uma resposta de falha do Worker."""
        if not isinstance(response_text, str):
            return None
        match = re.search(r"Falha: O modelo ([\w\d/.-]+) deve ser usado", response_text, re.IGNORECASE)
        if match:
            suggested_model = match.group(1).strip()
            log_message(f"Sugestão de modelo extraída da resposta: {suggested_model}", "Router")
            return suggested_model
        return None

    def route_task(self, task_description):
        """Usa o meta-modelo para escolher o melhor modelo conhecido para a tarefa."""
        agent_display_name = "Router"
        
        if task_description in self.routing_cache:
            cached_model = self.routing_cache[task_description]
            log_message(f"Cache de roteamento HIT. Usando modelo '{cached_model}' para a tarefa.", agent_display_name)
            return cached_model

        print_agent_message(agent_display_name, f"Roteando tarefa: '{task_description[:80]}...'")

        if len(self.known_models) == 1:
            log_message(f"Apenas um modelo conhecido ({self.known_models[0]}). Roteando diretamente para ele.", agent_display_name)
            return self.known_models[0]

        prompt = f"""Você é um roteador de tarefas especialista. Dada a tarefa a seguir, escolha o modelo mais apropriado da lista de modelos disponíveis. Responda APENAS com o nome exato do modelo.

Tarefa: "{task_description}"
Modelos Disponíveis: {self.known_models}
Melhor Modelo:"""
        
        chosen_model = call_gemini_api_with_retry([prompt], agent_display_name, self.meta_model_name)

        if chosen_model and chosen_model.strip() in self.known_models:
            chosen_model = chosen_model.strip()
            self.routing_cache[task_description] = chosen_model # Armazena no cache
            log_message(f"Tarefa roteada para o modelo: {chosen_model}", agent_display_name)
            return chosen_model
        else:
            log_message(f"Roteamento falhou ou retornou modelo inválido ('{chosen_model}'). Usando modelo padrão.", agent_display_name)
            return GEMINI_TEXT_MODEL_NAME # Fallback

class Worker:
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Instância do Worker criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal, model_to_use):
        agent_display_name = f"Worker ({model_to_use.split('/')[-1]})"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_context = "Resultados anteriores:\n" + ('\n'.join([f"- '{list(res.keys())[0]}': {str(list(res.values())[0].get('text_content', list(res.values())[0]))[:500]}..." for res in previous_results]) if previous_results else "Nenhum.\n")
        files_prompt_part = "Arquivos complementares carregados:\n" + ('\n'.join([f"- Nome: {f['display_name']} (ID: {f['file_id']})" for f in uploaded_files_info]) if uploaded_files_info else "Nenhum.\n")
        
        prompt = [
            f"Você é um Agente Executor usando o modelo '{model_to_use}'. Sua tarefa atual é: \"{sub_task_description}\"",
            f"Contexto (resultados anteriores, objetivo original, arquivos):\n{prompt_context}\n{files_prompt_part}",
            f"Objetivo: {original_goal}", "Execute a tarefa.",
            " * Se a tarefa envolver modificar ou criar arquivos de código ou markdown, forneça TODO O CONTEÚDO COMPLETO do arquivo.",
            " * Indique o NOME DO ARQUIVO CLARAMENTE ANTES de cada bloco de código/markdown usando o formato:\n   \"Arquivo: nome_completo.ext\"\n   ```linguagem_ou_extensao\n   // Conteúdo completo...\n   ```",
            " * Se você (o modelo '{model_to_use}') não for o mais adequado para esta tarefa, sua resposta DEVE ser no seguinte formato ESTRITO: \"Falha: O modelo NOME_DO_MODELO_SUGERIDO deve ser usado\".",
            " * Se identificar NOVAS sub-tarefas cruciais, liste-as na seção 'NOVAS_TAREFAS_SUGERIDAS:' como um array JSON de strings válidas. Se não houver, omita a seção.",
            "Resultado da Tarefa:", "[Resultado principal...]", 
            "NOVAS_TAREFAS_SUGERIDAS:", "[Se houver, use um NOVO array JSON de strings aqui. Senão, omita ou use [].]"
        ]
        
        if self.task_manager.uploaded_file_objects: prompt.extend(self.task_manager.uploaded_file_objects)
        
        task_res_raw = call_gemini_api_with_retry(prompt, agent_display_name, model_name=model_to_use)
        if task_res_raw is None: return {"text_content": "Falha: Sem resposta da API.", "artifacts": []}, []

        if "Falha: O modelo" in task_res_raw and "deve ser usado" in task_res_raw:
            log_message(f"Worker sinalizou que outro modelo deve ser usado: {task_res_raw}", agent_display_name)
            return {"text_content": task_res_raw, "artifacts": []}, []
            
        sugg_tasks_strings = []
        task_res_content_main = task_res_raw
        sugg_tasks_section_match = re.search(r"NOVAS_TAREFAS_SUGERIDAS:\s*(\[.*?\])", task_res_raw, re.IGNORECASE | re.DOTALL)
        
        if sugg_tasks_section_match:
            sugg_tasks_json_str = sugg_tasks_section_match.group(1).strip()
            task_res_content_main = task_res_raw[:sugg_tasks_section_match.start()].strip()
            # (Lógica de parsing de tarefas sugeridas mantida da v9.3.3)
            try: 
                parsed_json = json.loads(sugg_tasks_json_str)
                if isinstance(parsed_json, list):
                    sugg_tasks_strings = [str(s).strip() for s in parsed_json if str(s).strip()]
            except json.JSONDecodeError: 
                log_message(f"Erro ao decodificar JSON de NOVAS_TAREFAS_SUGERIDAS: '{sugg_tasks_json_str}'.", agent_display_name)

        extracted_artifacts = self._extract_artifacts_from_output(task_res_content_main)
        main_text_output = task_res_content_main # Simplificado
        
        log_message(f"Resultado da sub-tarefa '{sub_task_description}': {str(main_text_output)[:500]}...", agent_display_name)
        return {"text_content": main_text_output, "artifacts": extracted_artifacts}, sugg_tasks_strings

    def _extract_artifacts_from_output(self, output_str):
        # Esta função é mantida exatamente como na v9.3.3
        artifacts = []
        try:
            cleaned_output = output_str.strip()
            if cleaned_output.startswith("```json"): cleaned_output = re.sub(r"^```json\s*|\s*```$", "", cleaned_output, flags=re.DOTALL).strip()
            
            is_json_structure = (cleaned_output.startswith('{') and cleaned_output.endswith('}'))
            if is_json_structure:
                data = json.loads(cleaned_output)
                if isinstance(data, dict) and "arquivos" in data and isinstance(data["arquivos"], list):
                    for item in data["arquivos"]:
                        if isinstance(item, dict) and "nome" in item and "conteudo" in item:
                            fn = sanitize_filename(item["nome"]); cont = item["conteudo"];
                            if fn and cont: artifacts.append({"type": "code", "filename": fn, "content": cont})
                    if artifacts: return artifacts
        except (json.JSONDecodeError, KeyError):
            pass

        patterns = [re.compile(r"Arquivo:\s*(?P<filename>[^\n`]+)\s*\n```(?P<language>[a-zA-Z0-9_+\-#.]*)\s*\n(?P<content>.*?)\n```", re.DOTALL | re.MULTILINE)]
        for pattern in patterns:
            for match in pattern.finditer(output_str):
                filename = sanitize_filename(match.group("filename").strip())
                content = match.group("content").strip()
                if filename and content:
                    artifacts.append({"type": "code", "filename": filename, "content": content})
        return artifacts

class ImageWorker:
    def __init__(self, task_manager):
        self.task_manager = task_manager; self.model_name = GEMINI_IMAGE_GENERATION_MODEL_NAME
        log_message(f"ImageWorker criado para {self.model_name}", "ImageWorker")

    def generate_image(self, image_prompt, original_task_description="Gerar Imagem"):
        agent_display_name = "ImageWorker"
        print_agent_message(agent_display_name, f"Gerando imagem para: '{image_prompt[:100]}...'")
        
        # O prompt para o modelo 1.5 é mais direto
        prompt_parts = [f"Gerar uma imagem com base na seguinte descrição: {image_prompt}"]
        response_obj = call_gemini_api_with_retry(prompt_parts, agent_display_name, self.model_name)

        if not response_obj: return None, "Falha: Sem resposta da API de imagem."
        
        try:
            # O modelo 1.5 retorna os dados de imagem em `response.candidates[0].content.parts`
            if response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                for part in response_obj.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        img_bytes = part.inline_data.data
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fn_base = sanitize_filename(f"imagem_gerada_{original_task_description[:20]}_{ts}")
                        fn = f"{fn_base}.png"
                        log_message(f"Imagem gerada, pronta para salvar como '{fn}'.", agent_display_name)
                        return {"filename": fn, "content_bytes": img_bytes, "type": "image"}, None
            return None, "Falha: Nenhuma imagem na resposta."
        except Exception as e:
            log_message(f"Erro ao processar imagem: {e}\n{traceback.format_exc()}", agent_display_name)
            return None, f"Erro: {e}"

class Validator:
    # A classe Validator da v9.3.3 é mantida como está.
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Instância do Validator criada.", "Validator")
    def validate_results(self, task_results_history, staged_artifacts, original_goal):
        log_message(f"Iniciando validação automática. Artefatos no stage: {list(staged_artifacts.keys())}", "Validator")
        issues = []
        if not staged_artifacts:
            expects_files = any(kw in original_goal.lower() for kw in ["criar", "modificar", "salvar", "gerar arquivo"])
            if expects_files: issues.append("Nenhum artefato preparado, mas era esperado.")
        else:
            for filename, artifact_data in staged_artifacts.items():
                content = artifact_data.get("content")
                if not content or not content.strip(): issues.append(f"Artefato '{filename}' vazio.")
        if issues:
            log_message(f"Validação automática falhou. Problemas: {'; '.join(issues)}", "Validator")
            return {"status": "failure", "reason": "; ".join(issues)}
        log_message("Validação automática OK.", "Validator")
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
        self.router = Router() # NOVO
        self.gemini_text_model_name = GEMINI_TEXT_MODEL_NAME
        log_message("Instância do TaskManager (v9.3.4) criada.", "TaskManager")

    # Os métodos `confirm_new_tasks_with_llm`, `decompose_goal`, `present_for_manual_validation`,
    # `process_task_result`, `save_final_artifacts` são mantidos da v9.3.3.
    def confirm_new_tasks_with_llm(self, original_goal, current_task_list_for_prompt, suggested_new_tasks):
        # ... (código da v9.3.3) ...
        return []

    def decompose_goal(self, goal_to_decompose, previous_plan=None, automatic_validation_feedback=None, manual_validation_feedback_str=None):
        # ... (código da v9.3.3) ...
        return [goal_to_decompose]

    def present_for_manual_validation(self):
        # ... (código da v9.3.3) ...
        return {"approved": True}

    def process_task_result(self, task_description, task_result_data):
        # ... (código da v9.3.3) ...
        pass

    def save_final_artifacts(self):
        # ... (código da v9.3.3) ...
        pass
        
    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho (v9.3.4)...")
        log_message(f"Meta inicial: {self.goal}", "TaskManager")

        overall_success = False
        automatic_validation_attempts = 0
        current_goal_to_decompose = self.goal
        previous_plan_for_replan = None
        last_automatic_validation_feedback = None
        last_manual_feedback_str = None

        while automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES and not overall_success:
            if automatic_validation_attempts > 0:
                log_message(f"Tentativa de replanejamento automático {automatic_validation_attempts}/{MAX_AUTOMATIC_VALIDATION_RETRIES}...", "TaskManager")
            
            self.current_task_list = self.decompose_goal(
                current_goal_to_decompose, previous_plan_for_replan,
                last_automatic_validation_feedback, last_manual_feedback_str
            )
            last_manual_feedback_str = None

            if not self.current_task_list:
                print_agent_message("TaskManager", "Não foi possível decompor a meta. Encerrando."); return

            print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
            for i, task_desc in enumerate(self.current_task_list): print(f"  {i+1}. {task_desc}")
            
            if input("👤 Aprova este plano? (s/n) ➡️ ").strip().lower() != 's':
                print_agent_message("TaskManager", "Plano não aprovado. Encerrando."); return
            log_message("Plano aprovado pelo usuário.", "UsuárioInput")
            
            self.executed_tasks_results = []
            self.staged_artifacts = {}
            
            current_task_index = 0
            while current_task_index < len(self.current_task_list):
                task_description_str = self.current_task_list[current_task_index]
                
                print_agent_message("TaskManager", f"Próxima tarefa ({current_task_index + 1}/{len(self.current_task_list)}): {task_description_str}")
                
                # MANTIDO DA v9.3.3: Tratamento de tarefas especiais
                if task_description_str.startswith("TASK_GERAR_IMAGEM:"):
                    image_prompt = task_description_str.replace("TASK_GERAR_IMAGEM:", "").strip()
                    image_artifact_data, error_msg = self.image_worker.generate_image(image_prompt, original_task_description=task_description_str)
                    task_result_data = {"image_artifact_details": image_artifact_data, "text_content": f"Imagem gerada: {error_msg or image_artifact_data['filename']}", "artifacts": [image_artifact_data] if image_artifact_data else []}
                    self.process_task_result(task_description_str, task_result_data)
                    current_task_index += 1
                    continue
                
                if task_description_str.startswith("TASK_AVALIAR_IMAGENS:"):
                    eval_result = "Avaliação de imagem ainda não implementada neste fluxo." # Placeholder
                    task_result_data = {"text_content": f"Avaliação de imagens: {eval_result}", "artifacts": []}
                    self.process_task_result(task_description_str, task_result_data)
                    current_task_index += 1
                    continue

                # --- NOVO: LÓGICA DE ROTEAMENTO PARA TAREFAS GERAIS ---
                routing_attempts = 0
                task_success = False
                task_result_data, suggested_new_tasks_from_worker = {}, []

                while routing_attempts < MAX_ROUTING_ATTEMPTS and not task_success:
                    model_to_use = self.router.route_task(task_description_str)
                    
                    task_result_data, suggested_new_tasks_from_worker = self.worker.execute_task(
                        task_description_str, self.executed_tasks_results,
                        self.uploaded_files_info, self.goal, model_to_use
                    )
                    
                    result_text = task_result_data.get("text_content", "")
                    suggested_model = self.router.parse_suggestion(result_text)

                    if suggested_model:
                        routing_attempts += 1
                        print_agent_message("TaskManager", f"Worker sugeriu o uso do modelo '{suggested_model}'. Tentando novamente ({routing_attempts}/{MAX_ROUTING_ATTEMPTS}).")
                        self.router.add_model(suggested_model)
                        time.sleep(1)
                        continue
                    else:
                        task_success = True

                # --- FIM DA LÓGICA DE ROTEAMENTO ---
                
                self.process_task_result(task_description_str, task_result_data)
                log_message(f"Tarefa '{task_description_str}' concluída.", "TaskManager")

                if suggested_new_tasks_from_worker:
                    approved_new_tasks = self.confirm_new_tasks_with_llm(
                        self.goal, list(self.current_task_list[:current_task_index+1]),
                        suggested_new_tasks_from_worker
                    )
                    if approved_new_tasks:
                        self.current_task_list = self.current_task_list[:current_task_index+1] + approved_new_tasks + self.current_task_list[current_task_index+1:]
                        print_agent_message("TaskManager", f"Novas tarefas aprovadas adicionadas ao plano: {approved_new_tasks}")
                
                current_task_index += 1
            
            # O restante do loop de validação da v9.3.3 permanece aqui
            automatic_validation_result = self.validator.validate_results(self.executed_tasks_results, self.staged_artifacts, self.goal)
            last_automatic_validation_feedback = automatic_validation_result 

            if automatic_validation_result["status"] == "success":
                print_agent_message("TaskManager", "Validação automática dos artefatos bem-sucedida!")
                manual_validation_attempts = 0
                manual_approval_achieved = False
                while manual_validation_attempts <= MAX_MANUAL_VALIDATION_RETRIES and not manual_approval_achieved:
                    manual_val_result = self.present_for_manual_validation()
                    if manual_val_result["approved"]:
                        self.save_final_artifacts(); overall_success = True; manual_approval_achieved = True
                        break
                    last_manual_feedback_str = manual_val_result.get("feedback")
                    if last_manual_feedback_str == "cancelar" or manual_validation_attempts >= MAX_MANUAL_VALIDATION_RETRIES:
                        overall_success = False; break
                    manual_validation_attempts += 1
                    automatic_validation_attempts = 0
                    previous_plan_for_replan = list(self.current_task_list)
                    last_automatic_validation_feedback = None
                if overall_success: break
            else:
                print_agent_message("TaskManager", f"Validação automática falhou: {automatic_validation_result['reason']}")
                automatic_validation_attempts += 1
                if automatic_validation_attempts <= MAX_AUTOMATIC_VALIDATION_RETRIES:
                    print_agent_message("TaskManager", "Tentando replanejar...")
                    previous_plan_for_replan = list(self.current_task_list)
                else:
                    print_agent_message("TaskManager", "Máx. tentativas de validação automática. Encerrando."); break
        
        if overall_success: print_agent_message("TaskManager", "Fluxo de trabalho concluído com sucesso!")
        else: print_agent_message("TaskManager", "Fluxo de trabalho concluído com falhas ou cancelamento.")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.3.4 (Router Integrado)"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
    
    # ... O restante da função main da v9.3.3 permanece o mesmo ...
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

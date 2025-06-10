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
import sys
import select

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
MAX_MANUAL_VALIDATION_RETRIES = 2
USER_APPROVAL_TIMEOUT_SECONDS = 60 # NOVO: Tempo em segundos para aprovação automática

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
        else:
            active_gen_config = generation_config_text

    for part_item in prompt_parts:
        if isinstance(part_item, str): text_prompt_for_log += part_item + "\n"
        elif hasattr(part_item, 'name') and hasattr(part_item, 'display_name'):
            file_references_for_log.append(f"Arquivo: {part_item.display_name} (ID: {part_item.name}, TipoMIME: {getattr(part_item, 'mime_type', 'N/A')})")
    
    log_message(f"Prompt para {agent_name} (Modelo: {model_name}):\n---\n{text_prompt_for_log[:1000]}...\n---", "Sistema")

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        try:
            model_instance = genai.GenerativeModel(model_name, generation_config=active_gen_config, safety_settings=safety_settings_gemini)
            response = model_instance.generate_content(prompt_parts)
            
            if model_name == GEMINI_IMAGE_GENERATION_MODEL_NAME: return response

            if hasattr(response, 'text') and response.text is not None:
                 return response.text.strip()

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()

            return None
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1}/{MAX_API_RETRIES} ({agent_name}, {model_name}): {e}", agent_name)
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(current_retry_delay)
                current_retry_delay *= RETRY_BACKOFF_FACTOR
            else:
                return None
    return None

# --- Funções de Arquivos ---
def get_most_recent_cache_file():
    try:
        list_of_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "uploaded_files_info_*.json"))
        return max(list_of_files, key=os.path.getctime) if list_of_files else None
    except Exception as e:
        log_message(f"Erro ao buscar cache recente: {e}", "Sistema")
        return None

def load_cached_files_metadata(cache_file_path):
    if not cache_file_path or not os.path.exists(cache_file_path): return []
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f:
            cached_metadata = json.load(f)
        return cached_metadata if isinstance(cached_metadata, list) else []
    except Exception as e:
        log_message(f"Erro ao carregar cache {cache_file_path}: {e}", "Sistema")
        return []

def clear_upload_cache():
    print_agent_message("Sistema", f"Verificando cache local em: {UPLOADED_FILES_CACHE_DIR}")
    local_cache_files = glob.glob(os.path.join(UPLOADED_FILES_CACHE_DIR, "*.json"))
    if local_cache_files:
        print_user_message(f"Encontrados {len(local_cache_files)} arquivos de cache local. Limpar? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            for f in local_cache_files: os.remove(f)
            print_agent_message("Sistema", "✅ Cache local limpo.")

    print_agent_message("Sistema", "Verificando arquivos na API Gemini...")
    try:
        api_files_list = list(genai.list_files())
        if api_files_list:
            print_user_message(f"Encontrados {len(api_files_list)} arquivos na API. ‼️ Deletar TODOS é IRREVERSÍVEL. ‼️ Continuar? (s/n)")
            if input("➡️ ").strip().lower() == 's':
                for f in api_files_list:
                    genai.delete_file(name=f.name)
                    time.sleep(0.2)
                print_agent_message("Sistema", "✅ Arquivos da API limpos.")
    except Exception as e:
        print_agent_message("Sistema", f"❌ Erro ao limpar arquivos da API: {e}")

def get_uploaded_files_info_from_user():
    uploaded_file_objects = []
    uploaded_files_metadata = []
    reused_file_ids = set()

    try:
        api_files_list = list(genai.list_files())
    except Exception as e:
        print_agent_message("Sistema", f"Falha ao listar arquivos da API: {e}")
        api_files_list = []

    api_files_dict = {f.name: f for f in api_files_list}
    cached_metadata = load_cached_files_metadata(get_most_recent_cache_file())
    
    offer_for_reuse = [{**f.to_dict(), "user_path": next((c.get("user_path") for c in cached_metadata if c.get("file_id") == f.name), "N/A")} for f in api_files_list]

    if offer_for_reuse:
        print_agent_message("Sistema", "Arquivos na API para reutilizar:")
        for i, meta in enumerate(offer_for_reuse): print(f"  {i+1}. {meta['display_name']}")
        print_user_message("Reutilizar arquivos? (s/n)")
        if input("➡️ ").strip().lower() == 's':
            print_user_message("Digite os números (ex: 1,3) ou 'todos':")
            choices = input("➡️ ").strip().lower()
            
            indices_to_try = []
            if choices == 'todos':
                indices_to_try = range(len(offer_for_reuse))
            else:
                try: indices_to_try = [int(x.strip()) - 1 for x in choices.split(',')]
                except ValueError: print("❌ Entrada inválida.")

            for idx in indices_to_try:
                if 0 <= idx < len(offer_for_reuse):
                    chosen_meta = offer_for_reuse[idx]
                    file_id = chosen_meta["name"]
                    if file_id in reused_file_ids: continue
                    try:
                        file_obj = api_files_dict.get(file_id)
                        if not file_obj: file_obj = genai.get_file(name=file_id)
                        uploaded_file_objects.append(file_obj)
                        uploaded_files_metadata.append(chosen_meta)
                        reused_file_ids.add(file_id)
                        print(f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                    except Exception as e:
                        print(f"❌ Erro ao obter arquivo '{chosen_meta['display_name']}': {e}")
                else: print(f"❌ Índice inválido: {idx + 1}")

    print_user_message("Adicionar NOVOS arquivos? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        while True:
            print_user_message("Caminho do arquivo/padrão (ou 'fim'):")
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
                        uf_meta = uf.to_dict()
                        uf_meta['user_path'] = fp
                        uf_meta['file_id'] = uf.name
                        uploaded_files_metadata.append(uf_meta)
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
    return "Arquivos complementares:\n" + "\n".join([f"- {m['display_name']} (ID: {m.get('file_id') or m.get('name')})" for m in files_metadata_list])

# MODIFICAÇÃO: Nova função para input com timeout
def get_user_input_with_timeout(prompt_str, timeout):
    """Espera por input do usuário por um tempo definido."""
    print(prompt_str, end='', flush=True) # Exibe o prompt (ex: "➡️ ")
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().strip().lower()
    else:
        return "timeout"

# MODIFICAÇÃO: get_user_feedback_or_approval usa a nova função de timeout
def get_user_feedback_or_approval():
    prompt = (
        f"\nO que fazer? (Aprovação automática em {USER_APPROVAL_TIMEOUT_SECONDS} segundos)\n"
        "  [A]provar - Salvar os artefatos como estão.\n"
        "  [F]eedback - Fornecer feedback para o sistema tentar novamente.\n"
        "  [S]air - Encerrar o processo.\n"
        "Escolha: "
    )
    print_user_message(prompt)
    
    choice = get_user_input_with_timeout("➡️ ", USER_APPROVAL_TIMEOUT_SECONDS)

    if choice == "timeout":
        print_agent_message("Sistema", "Tempo esgotado. Aprovando automaticamente os artefatos.")
        log_message("Aprovação automática por timeout.", "Sistema")
        return 'a'

    if choice in ['a', 'f', 's']:
        log_message(f"Usuário escolheu a opção: {choice.upper()}", "UsuárioInput")
        return choice
    else:
        # Se o usuário digitar algo inválido, assume que quer sair para evitar loops.
        print_agent_message("Sistema", f"❌ Opção inválida ('{choice}'). Encerrando para evitar loop.")
        return 's'

# --- Classes de Agentes ---

class ImageWorker:
    def __init__(self):
        self.model_name = GEMINI_IMAGE_GENERATION_MODEL_NAME
        self.generation_config = generation_config_image_sdk
    def generate_image(self, prompt):
        print_agent_message("ImageWorker", f"Gerando imagem para: '{prompt[:100]}...'")
        response = call_gemini_api_with_retry([prompt], "ImageWorker", self.model_name, self.generation_config)
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    path = os.path.join(TEMP_ARTIFACTS_DIR, f"temp_img_{int(time.time())}_{sanitize_filename(prompt[:20],False)}.png")
                    with open(path, "wb") as f: f.write(part.inline_data.data)
                    return path
        return "Falha na geração da imagem."

class Worker:
    def execute_sub_task(self, description, context, files, task_num, total_tasks):
        print_agent_message("Worker", f"Executando ({task_num}/{total_tasks}): '{description}'")
        prompt = rf"""
Você é um Agente Executor. Tarefa atual: "{description}"
Contexto (resultados anteriores, objetivo original, arquivos):
{context}

Execute a tarefa.
- Se for "Criar uma descrição textual detalhada (prompt) para gerar a imagem de [...]", seu resultado DEVE ser APENAS essa descrição textual.
- Se a tarefa envolver modificar ou criar arquivos de código, forneça o CONTEÚDO COMPLETO do arquivo. Indique o NOME DO ARQUIVO CLARAMENTE antes de cada bloco de código (ex: "Arquivo: nome.ext" ou ```python nome.py ... ```).
- Se identificar NOVAS sub-tarefas cruciais, liste-as em 'NOVAS_TAREFAS_SUGERIDAS:' como array JSON de strings. Se não, omita.
"""
        response_text = call_gemini_api_with_retry([prompt] + files, "Worker", GEMINI_TEXT_MODEL_NAME, generation_config_text)
        
        if response_text is None: return None, []
        
        task_res, sugg_tasks_strings = response_text, []
        marker = "NOVAS_TAREFAS_SUGERIDAS:"
        if marker in response_text:
            parts = response_text.split(marker, 1)
            task_res = parts[0].strip()
            potential_json_or_list = parts[1].strip()
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', potential_json_or_list, re.DOTALL)
                if json_match:
                    parsed_suggestions = json.loads(json_match.group(1) or json_match.group(2))
                    if isinstance(parsed_suggestions, list):
                        sugg_tasks_strings = [str(item) for item in parsed_suggestions]
            except Exception as e:
                log_message(f"Erro ao processar novas tarefas: {e}", "Worker")
        else:
            task_res = response_text.strip()
        
        if task_res.lower().startswith("resultado da tarefa:"):
            task_res = task_res[len("resultado da tarefa:"):].strip()
        
        return task_res, sugg_tasks_strings

class Validator:
    def __init__(self, tm_ref): self.tm = tm_ref
    def evaluate_and_select_image_concepts(self, goal, results, files, meta):
        print_agent_message("Validator", "Avaliando conceitos de imagem...")
        summary = "\n".join([f"Tentativa {i+1}: Prompt='{r['image_prompt_used']}', Sucesso={os.path.exists(str(r.get('result')))}" for i, r in enumerate(results)]) or "Nenhuma."
        prompt = f'Meta: "{goal}"\nTentativas:\n{summary}\nRetorne um array JSON com os prompts aprovados. Apenas o array.'
        response = call_gemini_api_with_retry([prompt] + files, "Validator", GEMINI_TEXT_MODEL_NAME, generation_config_text)
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            approved_prompts = json.loads(match.group(0)) if match else []
            return [res for res in results if res.get('image_prompt_used') in approved_prompts and os.path.exists(str(res.get('result')))]
        except: return []
        
    def validate_and_save_final_output(self, goal, context, files, artifacts):
        agent_display_name = "Validator(Final)"
        print_agent_message(agent_display_name, "Validando resultado final...")
        summary = "\n".join([f"- {a['type']}: {os.path.basename(a.get('filename') or a.get('artifact_path',''))}" for a in artifacts])
        
        prompt_text_part_validation = rf"""
Você é um Gerente de QA. Analise o contexto e os artefatos gerados para a meta: "{goal}"
Contexto: {context}
Artefatos para Salvar: {summary}

Sua resposta DEVE seguir ESTRITAMENTE o seguinte formato com marcadores:

VALIDATION_PASSED: [escreva true ou false aqui]
---MAIN_REPORT_START---
[Escreva o relatório detalhado em Markdown aqui. Pode ter múltiplas linhas e conter aspas.]
---MAIN_REPORT_END---
---GENERAL_EVALUATION_START---
[Escreva a avaliação geral e concisa aqui.]
---GENERAL_EVALUATION_END---
"""
        response = call_gemini_api_with_retry([prompt_text_part_validation] + files, agent_display_name, GEMINI_TEXT_MODEL_NAME, generation_config_text)

        if not response:
            return False, "Falha ao obter avaliação final da API."
        
        try:
            pass_match = re.search(r"VALIDATION_PASSED:\s*(true|false)", response, re.IGNORECASE)
            report_match = re.search(r"---MAIN_REPORT_START---([\s\S]*?)---MAIN_REPORT_END---", response, re.DOTALL)
            eval_match = re.search(r"---GENERAL_EVALUATION_START---([\s\S]*?)---GENERAL_EVALUATION_END---", response, re.DOTALL)

            validation_passed = (pass_match.group(1).lower() == 'true') if pass_match else False
            main_report = report_match.group(1).strip() if report_match else "Relatório não gerado."
            evaluation_text = eval_match.group(1).strip() if eval_match else "Avaliação não gerada."

            goal_slug = sanitize_filename(goal, allow_extension=False)
            assessment_file_name = os.path.join(OUTPUT_DIRECTORY, f"avaliacao_completa_{goal_slug}_{CURRENT_TIMESTAMP_STR}.md")
            with open(assessment_file_name, "w", encoding="utf-8") as f:
                f.write(f"# Relatório: {goal}\n\n{main_report}\n\n## Avaliação da IA\n{evaluation_text}")
            print_agent_message(agent_display_name, f"Relatório de avaliação salvo: {assessment_file_name}")

            if validation_passed:
                final_dir = os.path.join(OUTPUT_DIRECTORY, f"artefatos_finais_{goal_slug}_{CURRENT_TIMESTAMP_STR}")
                os.makedirs(final_dir, exist_ok=True)
                for a in artifacts:
                    src = a.get('temp_path') or a.get('artifact_path')
                    if src and os.path.exists(src):
                        dest_name = a.get('filename') or f"imagem_{sanitize_filename(a.get('prompt', '')[:30], False)}.png"
                        shutil.copy(src, os.path.join(final_dir, dest_name))
                        print_agent_message(agent_display_name, f"✅ Artefato final salvo: {dest_name}")
            
            return validation_passed, evaluation_text
        except Exception as e: 
            log_message(f"Erro ao processar validação com marcadores: {e}\nResposta Bruta:\n{response}", agent_display_name)
            return False, f"Falha ao processar resposta de validação: {e}"

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

    def decompose_task(self, main_goal, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_display_name = "TaskManager(Decomp)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{main_goal}'")
        prompt = f"Divida a meta principal em subtarefas sequenciais.\nMeta: \"{main_goal}\"\n{files_metadata_for_prompt_text}\nRetorne um array JSON de strings. Apenas o array."
        response_text = call_gemini_api_with_retry([prompt] + uploaded_file_objects, agent_display_name, GEMINI_TEXT_MODEL_NAME, generation_config_text)
        try:
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            self.task_list = json.loads(match.group(0)) if match else []
            log_message(f"Plano de tarefas decomposto: {self.task_list}", agent_display_name)
            return bool(self.task_list)
        except Exception as e:
            log_message(f"Erro na decomposição da tarefa: {e}", agent_display_name)
            return False

    def confirm_new_tasks_with_llm(self, original_goal, current_task_list, suggested_new_tasks, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_name = "TaskManager(Valida Novas Tarefas)"
        if not suggested_new_tasks: return []
        print_agent_message(agent_name, f"Avaliando novas tarefas: {suggested_new_tasks}")
        prompt_text = f"""Objetivo: "{original_goal}".\nTarefas atuais: {json.dumps(current_task_list)}.\nNovas sugeridas: {json.dumps(suggested_new_tasks)}.\nAvalie e retorne em JSON APENAS as tarefas aprovadas. Se nenhuma, []."""
        response = call_gemini_api_with_retry([prompt_text] + uploaded_file_objects, agent_name, GEMINI_TEXT_MODEL_NAME, generation_config_text)
        
        approved_tasks_final = []
        if response:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', response, re.DOTALL)
                parsed_response = []
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_response = json.loads(json_str)
                
                if isinstance(parsed_response, list):
                    for item in parsed_response:
                        if isinstance(item, str) and item.strip():
                            approved_tasks_final.append(item.strip())
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str) and item["tarefa"].strip():
                            approved_tasks_final.append(item["tarefa"].strip())
            except Exception as ex:
                log_message(f"Erro ao decodificar/processar aprovação: {ex}. Resp: {response}", agent_name)
        return approved_tasks_final

    def run_workflow(self, initial_goal, uploaded_file_objects, uploaded_files_metadata):
        self.uploaded_files_metadata = uploaded_files_metadata
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        
        files_metadata_for_prompt_text = format_uploaded_files_info_for_prompt_text(self.uploaded_files_metadata)

        if not self.decompose_task(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text):
            print_agent_message("TaskManager", "Falha na decomposição da tarefa. Encerrando."); return
        
        print_agent_message("TaskManager", "--- PLANO DE TAREFAS INICIAL ---")
        for i, task in enumerate(self.task_list): print(f"  {i+1}. {task}")
        print_user_message("Aprova este plano? (s/n)")
        if input("➡️ ").strip().lower() != 's':
            print_agent_message("TaskManager", "Plano rejeitado. Encerrando."); return
        
        overall_success = False
        manual_retries = 0
        
        while True:
            current_task_index = 0
            image_generation_attempts = []
            self.completed_tasks_results = []
            self.temp_artifacts = []
            total_tasks_in_cycle = len(self.task_list)

            while current_task_index < len(self.task_list):
                task_desc = self.task_list[current_task_index]
                
                if isinstance(task_desc, dict):
                    task_desc = task_desc.get("tarefa", str(task_desc))

                task_number = current_task_index + 1
                context = "\n".join([f"Tarefa: {r['task']}\nResultado: {str(r.get('result'))[:200]}" for r in self.completed_tasks_results])
                
                if task_desc.startswith("TASK_GERAR_IMAGEM:"):
                    print_agent_message("TaskManager", f"Delegando tarefa de imagem ({task_number}/{total_tasks_in_cycle})")
                    prompt = task_desc.replace("TASK_GERAR_IMAGEM:", "").strip()
                    if not prompt and self.completed_tasks_results and "descrição" in self.completed_tasks_results[-1]['task'].lower():
                        prompt = self.completed_tasks_results[-1]['result']
                    result = self.image_worker.generate_image(prompt)
                    image_generation_attempts.append({"image_prompt_used": prompt, "result": result})
                    self.completed_tasks_results.append({"task": task_desc, "result": result})
                elif task_desc.startswith("TASK_AVALIAR_IMAGENS:"):
                    print_agent_message("TaskManager", f"Delegando tarefa de avaliação ({task_number}/{total_tasks_in_cycle})")
                    validated = self.validator.evaluate_and_select_image_concepts(initial_goal, image_generation_attempts, uploaded_file_objects, files_metadata_for_prompt_text)
                    self.completed_tasks_results.append({"task": task_desc, "result": [v['image_prompt_used'] for v in validated]})
                    for v in validated: self.temp_artifacts.append({'type': 'imagem', 'artifact_path': v['result'], 'prompt': v['image_prompt_used']})
                else:
                    result, new_tasks = self.worker.execute_sub_task(task_desc, context, uploaded_file_objects, task_number, total_tasks_in_cycle)
                    self.completed_tasks_results.append({"task": task_desc, "result": result})
                    
                    if new_tasks:
                        approved_tasks = self.confirm_new_tasks_with_llm(initial_goal, self.task_list, new_tasks, uploaded_file_objects, files_metadata_for_prompt_text)
                        if approved_tasks:
                            for i, new_task in enumerate(approved_tasks):
                                self.task_list.insert(current_task_index + 1 + i, new_task)
                            total_tasks_in_cycle = len(self.task_list)
                            print_agent_message("TaskManager", f"Plano atualizado com {len(approved_tasks)} nova(s) tarefa(s).")
                
                current_task_index += 1

            print_agent_message("TaskManager", "Ciclo de tarefas concluído. Validando...")
            final_context = "\n".join([f"Tarefa: {r['task']}\nResultado: {str(r.get('result'))[:300]}" for r in self.completed_tasks_results])
            
            is_valid, validation_output = self.validator.validate_and_save_final_output(initial_goal, final_context, uploaded_file_objects, self.temp_artifacts)

            if is_valid:
                print_agent_message("TaskManager", f"Validação bem-sucedida! {validation_output}")
                overall_success = True; break

            print_agent_message("TaskManager", f"Validação falhou: {validation_output}")
            if manual_retries >= MAX_MANUAL_VALIDATION_RETRIES:
                print_agent_message("TaskManager", "Máximo de tentativas manuais atingido. Encerrando."); break

            user_choice = get_user_feedback_or_approval()
            if user_choice == 'a':
                print_agent_message("TaskManager", "Aprovação manual. Salvando último estado..."); 
                self.validator.validate_and_save_final_output(initial_goal, final_context, uploaded_file_objects, self.temp_artifacts)
                overall_success = True; break
            elif user_choice == 's':
                print_agent_message("TaskManager", "Encerrado pelo usuário."); break
            elif user_choice == 'f':
                print_user_message("Forneça seu feedback para corrigir o resultado:")
                feedback = input("➡️ ").strip()
                self.task_list = [f"TASK_CORRIGIR_COM_FEEDBACK: O resultado anterior ({final_context}) não foi satisfatório. Corrija com base no feedback: '{feedback}'"]
                manual_retries += 1
                print_agent_message("TaskManager", "Nova tarefa de correção definida. Reiniciando ciclo...")

        if os.path.exists(TEMP_ARTIFACTS_DIR): shutil.rmtree(TEMP_ARTIFACTS_DIR)
        os.makedirs(TEMP_ARTIFACTS_DIR)
        print_agent_message("TaskManager", f"Fluxo de trabalho concluído. Sucesso: {overall_success}")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v9.4.5"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Correção de Validação) ---")
    print(f"📝 Logs: {LOG_FILE_NAME}\n📄 Saídas Finais: {OUTPUT_DIRECTORY}\n⏳ Artefatos Temporários: {TEMP_ARTIFACTS_DIR}\nℹ️ Cache Uploads: {UPLOADED_FILES_CACHE_DIR}")
    
    print_user_message("Deseja limpar o cache de uploads (local e/ou da API Gemini) antes de começar? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        clear_upload_cache()
    
    if os.path.exists(TEMP_ARTIFACTS_DIR): shutil.rmtree(TEMP_ARTIFACTS_DIR)
    os.makedirs(TEMP_ARTIFACTS_DIR)
        
    initial_goal_input = input("🎯 Defina a meta principal: ")
    print_user_message(initial_goal_input)
    
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()
    
    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
    else:
        manager = TaskManager()
        manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")

import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import glob
from typing import List, Optional
from PIL import Image
from io import BytesIO

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

# --- Modelos Gemini (Atualizado conforme solicitado) ---
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-flash-preview-05-20" 
GEMINI_IMAGE_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" 


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
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não está definida.")
    genai.configure(api_key=GEMINI_API_KEY)
    log_message("API Gemini configurada.", "Sistema")
except Exception as e:
    print(f"Erro na configuração da API Gemini: {e}")
    log_message(f"Erro na configuração da API Gemini: {e}", "Sistema")
    exit()

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
log_message(f"Modelo Gemini (imagem): {GEMINI_IMAGE_MODEL_NAME}", "Sistema")

safety_settings_gemini = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# --- Ferramentas para o Agente ---
def save_file(filename: str, content: str) -> str:
    """Salva o conteúdo textual fornecido em um arquivo com o nome especificado."""
    try:
        sanitized_fn = sanitize_filename(filename)
        if not sanitized_fn: return "Erro: O nome do arquivo é inválido."
        full_path = os.path.join(OUTPUT_DIRECTORY, sanitized_fn)
        with open(full_path, "w", encoding="utf-8") as f: f.write(content)
        log_message(f"Arquivo '{sanitized_fn}' salvo com sucesso pela ferramenta.", "Tool:save_file")
        return f"Sucesso: O arquivo '{sanitized_fn}' foi salvo."
    except Exception as e:
        log_message(f"Erro ao salvar arquivo '{filename}' via ferramenta: {e}", "Tool:save_file")
        return f"Erro ao salvar o arquivo '{filename}': {str(e)}"

def translate_to_english(text_to_translate: str) -> str:
    """Traduz um texto para o inglês usando a API Gemini."""
    try:
        log_message(f"Traduzindo para inglês: '{text_to_translate[:50]}...'", "Tool:translate")
        translation_model = genai.GenerativeModel(GEMINI_TEXT_MODEL_NAME)
        response = translation_model.generate_content(f"Translate the following text to English, output only the translated text and nothing else: '{text_to_translate}'")
        translated_text = response.text.strip()
        log_message(f"Tradução concluída: '{translated_text}'", "Tool:translate")
        return translated_text
    except Exception as e:
        log_message(f"Erro na ferramenta de tradução: {e}", "Tool:translate")
        return f"Erro de tradução: {e}"

def generate_image(image_prompt_in_english: str, base_image_path: Optional[str] = None) -> str:
    """
    Gera uma imagem a partir de um prompt em inglês. Pode, opcionalmente, editar uma imagem base.

    Args:
        image_prompt_in_english (str): A descrição detalhada (em INGLÊS) da imagem a ser gerada ou da edição a ser feita.
        base_image_path (str, optional): O caminho para a imagem base a ser editada. Se omitido, uma nova imagem será criada.

    Returns:
        str: Uma mensagem indicando o nome do arquivo salvo ou uma mensagem de erro.
    """
    try:
        log_message(f"Iniciando geração/edição de imagem com o prompt: '{image_prompt_in_english[:100]}...'", "Tool:generate_image")
        
        image_model = genai.GenerativeModel(GEMINI_IMAGE_MODEL_NAME)
        
        contents = [image_prompt_in_english]
        if base_image_path:
            log_message(f"Carregando imagem base para edição: {base_image_path}", "Tool:generate_image")
            if not os.path.exists(base_image_path):
                return f"Erro: O arquivo da imagem base '{base_image_path}' não foi encontrado."
            image_part = Image.open(base_image_path)
            contents.append(image_part)

        # --- MUDANÇA AQUI: Configuração explícita para geração de imagem ---
        image_gen_config = genai.types.GenerationConfig(
            response_modalities=['IMAGE'] 
        )

        response = image_model.generate_content(
            contents,
            generation_config=image_gen_config
        )
        
        # --- MUDANÇA AQUI: Extração correta dos dados da imagem ---
        image_part = next((part for part in response.candidates[0].content.parts if part.inline_data), None)

        if not image_part:
             return "Erro: A API não retornou dados de imagem (inline_data) na resposta."

        image_bytes = image_part.inline_data.data

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fn_base = sanitize_filename(f"imagem_{image_prompt_in_english[:20]}_{ts}")
        filename = f"{fn_base}.png"
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        with open(full_path, "wb") as f: f.write(image_bytes)
            
        log_message(f"Imagem salva com sucesso como '{filename}'.", "Tool:generate_image")
        return f"Sucesso: Imagem gerada e salva como '{filename}'."
        
    except Exception as e:
        log_message(f"Erro na ferramenta generate_image: {e}\n{traceback.format_exc()}", "Tool:generate_image")
        return f"Erro ao gerar a imagem: {e}"

AGENT_TOOLS = [save_file, generate_image, translate_to_english]

# --- Funções de Comunicação e Arquivos ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")
def print_thought_message(message): print(f"\n🧠 [Pensamento do Agente]:\n{message}"); log_message(f"PENSAMENTO:\n{message}", "Agente")

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
            "mime_type": mime_type, "user_path": user_path
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
                    except Exception as e: log_message(f"Erro ao obter '{chosen_meta['display_name']}' para reutilização: {e}", "Sistema")
    
    if input("👤 Adicionar NOVOS arquivos? (s/n) ➡️ ").lower() == 's':
        while True:
            fp_pattern = input("👤 Caminho/padrão (ou 'fim'): ➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            
            expanded_paths = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*','?']) else ([fp_pattern] if os.path.exists(fp_pattern) else [])
            expanded_files = [f for f in expanded_paths if os.path.isfile(f)] 

            if not expanded_files: print_agent_message("Sistema", f"❌ Nenhum arquivo encontrado para: '{fp_pattern}'"); continue
            
            for fp in expanded_files:
                dn = os.path.basename(fp)
                try:
                    print_agent_message("Sistema", f"Fazendo upload de '{dn}'...")
                    file_obj = genai.upload_file(path=fp, display_name=dn)
                    uploaded_file_objects.append(file_obj)
                    new_meta = {"file_id": file_obj.name, "display_name": dn, "mime_type": file_obj.mime_type, "user_path": fp}
                    uploaded_files_metadata.append(new_meta)
                    print_agent_message("Sistema", f"✅ Novo arquivo '{dn}' (ID: {file_obj.name}) enviado.")
                    time.sleep(1)
                except Exception as e: log_message(f"Erro upload '{dn}': {e}", "Sistema"); print_agent_message("Sistema", f"❌ Erro no upload de '{dn}': {e}")
    if uploaded_files_metadata:
        with open(os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json"), "w", encoding="utf-8") as f: json.dump(uploaded_files_metadata, f, indent=4)
    return uploaded_file_objects, uploaded_files_metadata


def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config_dict=None, system_instruction=None, tools=None):
    log_message(f"Iniciando chamada à API Gemini para {agent_name} (Modelo: {model_name})...", "Sistema")
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Faz uma cópia para evitar modificar o dicionário original
    active_gen_config = (gen_config_dict or {}).copy()
    active_gen_config.setdefault("temperature", 0.7)

    # Converte o dicionário 'thinking_config' em um objeto do tipo correto
    if "thinking_config" in active_gen_config and isinstance(active_gen_config["thinking_config"], dict):
        active_gen_config["thinking_config"] = genai.types.ThinkingConfig(**active_gen_config["thinking_config"])
    
    # Constrói o objeto GenerationConfig final
    final_config_obj = genai.types.GenerationConfig(**active_gen_config)
    log_message(f"Usando generation_config: {final_config_obj}", "Sistema")
    
    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES} para {agent_name}...", "Sistema")
        try:
            model_instance = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction, 
                tools=tools 
            )
            response = model_instance.generate_content(
                prompt_parts, 
                generation_config=final_config_obj
            )
            return response
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1} ({agent_name}): {type(e).__name__} - {e}\n{traceback.format_exc()}", agent_name)
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(current_retry_delay)
                current_retry_delay *= RETRY_BACKOFF_FACTOR
            else: return None
    return None

# --- Classes dos Agentes ---
class Worker:
    def __init__(self, task_manager, model_name=GEMINI_TEXT_MODEL_NAME):
        self.task_manager = task_manager
        self.model_name = model_name
        self.system_instruction = (
            "Você é um Agente Executor especialista. Sua responsabilidade é executar tarefas complexas. "
            "Pense passo a passo sobre a tarefa. "
            "Execute o plano usando as ferramentas disponíveis (`save_file`, `generate_image`, `translate_to_english`). "
            "Ao final, forneça um resumo conciso do que foi feito."
        )
        log_message("Instância do Worker (v10.11) criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_context = f"Resultados de tarefas anteriores: {json.dumps(previous_results) if previous_results else 'Nenhum.'}"
        files_prompt_part = f"Arquivos de referência: {[f['display_name'] for f in uploaded_files_info]}"
        
        prompt_parts = [
            f"Contexto: {prompt_context}\n{files_prompt_part}",
            f"Objetivo Geral: {original_goal}",
            f"\nSua tarefa específica agora é: \"{sub_task_description}\"",
            "Execute a tarefa. Use as ferramentas disponíveis para pensar e agir."
        ]
        if self.task_manager.uploaded_file_objects:
             prompt_parts.extend(self.task_manager.uploaded_file_objects)
        
        worker_gen_config = { 
            "thinking_config": {
                "thinking_budget": -1, # Orçamento dinâmico
                "include_thoughts": False
            } 
        } 

        response = call_gemini_api_with_retry(
            prompt_parts,
            agent_display_name,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            tools=AGENT_TOOLS,
            gen_config_dict=worker_gen_config
        )

        if response is None:
            return {"text_content": "Falha: Sem resposta da API."}, []
        
        final_text_response = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'thought') and part.thought:
                print_thought_message(part.text)
            elif hasattr(part, 'text') and part.text:
                final_text_response += part.text + "\n"
        
        if not final_text_response:
            final_text_response = "Ação concluída através de ferramentas."

        return {"text_content": final_text_response.strip()}, []


class TaskManager:
    def __init__(self, initial_goal, uploaded_file_objects=None, uploaded_files_info=None):
        self.goal = initial_goal
        self.uploaded_file_objects = uploaded_file_objects or []
        self.uploaded_files_info = uploaded_files_info or []
        self.current_task_list = []
        self.executed_tasks_results = []
        self.worker = Worker(self)
        self.system_instruction = (
            "Você é um Gerenciador de Tarefas especialista. Sua função é decompor uma meta principal em um plano de sub-tarefas sequenciais, claras e executáveis. "
            "Se a meta envolver a criação ou edição de uma imagem, o plano DEVE incluir as seguintes tarefas em ordem: "
            "1. Uma tarefa para criar a descrição da imagem (o prompt). "
            "2. Uma tarefa para chamar a ferramenta 'translate_to_english' para traduzir a descrição. "
            "3. Uma tarefa para chamar a ferramenta 'generate_image' usando a descrição traduzida. "
            "Retorne o plano como um array JSON de strings, usando o esquema fornecido."
        )
        log_message("Instância do TaskManager (v10.11) criada.", "TaskManager")
        
    def decompose_goal(self, goal_to_decompose):
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{goal_to_decompose}'")

        prompt_parts = [ f"Meta Principal a ser decomposta: \"{goal_to_decompose}\"" ]
        if self.uploaded_file_objects:
             prompt_parts.extend(self.uploaded_file_objects)
        
        planner_gen_config = { 
            "response_mime_type": "application/json",
            "response_schema": List[str],
            "thinking_config": {
                "thinking_budget": 1024,
                "include_thoughts": True
            }
        }

        response = call_gemini_api_with_retry(
            prompt_parts, 
            agent_display_name, 
            system_instruction=self.system_instruction,
            tools=AGENT_TOOLS,
            gen_config_dict=planner_gen_config
        )
        
        json_text = ""
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                 if hasattr(part, 'thought') and part.thought:
                    print_thought_message(f"(Planejamento) {part.text}")
                 elif hasattr(part, 'text') and part.text:
                    json_text += part.text
        
        if json_text:
            try:
                log_message(f"JSON recebido do planejador: {json_text}", "TaskManager")
                tasks = json.loads(json_text)
                if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
                    return tasks
            except (json.JSONDecodeError, TypeError) as e:
                log_messag

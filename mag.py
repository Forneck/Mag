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

# --- Modelos Gemini (Atualizado para estabilidade e capacidade multimodal) ---
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
    # CLIENT = genai.Client() # Removido para corrigir AttributeError
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
@genai.tool
def save_file(filename: str, content: str) -> str:
    """Salva o conteúdo textual fornecido em um arquivo com o nome especificado."""
    try:
        sanitized_fn = sanitize_filename(filename)
        if not sanitized_fn: return "Erro: O nome do arquivo é inválido."
        full_path = os.path.join(OUTPUT_DIRECTORY, sanitized_fn)
        with open(full_path, "w", encoding="utf-8") as f: f.write(content)
        log_message(f"Arquivo '{sanitized_fn}' salvo com sucesso.", "Tool:save_file")
        return f"Sucesso: O arquivo '{sanitized_fn}' foi salvo."
    except Exception as e:
        log_message(f"Erro ao salvar arquivo '{filename}': {e}", "Tool:save_file")
        return f"Erro ao salvar o arquivo '{filename}': {str(e)}"

@genai.tool
def create_cache(file_ids: List[str], system_instruction: Optional[str] = None, ttl_seconds: int = 3600) -> str:
    """
    Cria um cache de conteúdo explícito a partir de arquivos e uma instrução de sistema para reduzir custos em chamadas futuras.

    Args:
        file_ids (List[str]): Uma lista de IDs de arquivos (obtidos ao fazer upload) para incluir no cache.
        system_instruction (str, optional): Uma instrução de sistema longa ou repetitiva para incluir no cache.
        ttl_seconds (int, optional): O tempo de vida do cache em segundos. Padrão de 1 hora (3600s).

    Returns:
        str: O nome do cache criado (ex: 'cachedContents/xxxx') ou uma mensagem de erro.
    """
    try:
        log_message(f"Criando cache com arquivos: {file_ids} e TTL: {ttl_seconds}s.", "Tool:create_cache")
        # Usa genai.get_file, pois o cliente foi removido
        contents = [genai.get_file(name=fid) for fid in file_ids]
        
        # O SDK pode não suportar client.caches.create diretamente.
        # Esta funcionalidade pode precisar de uma abordagem diferente ou pode estar obsoleta.
        # Por enquanto, esta ferramenta provavelmente falhará.
        # Requereria um cliente gRPC para acesso direto.
        return "Erro: A criação de cache explícito via esta ferramenta não é suportada no momento."

    except Exception as e:
        log_message(f"Erro na ferramenta create_cache: {e}\n{traceback.format_exc()}", "Tool:create_cache")
        return f"Erro ao criar o cache: {e}"


@genai.tool
def generate_image(image_prompt_in_english: str, base_image_path: Optional[str] = None) -> str:
    """Gera ou edita uma imagem a partir de um prompt em inglês."""
    try:
        log_message(f"Iniciando geração/edição de imagem com o prompt: '{image_prompt_in_english[:100]}...'", "Tool:generate_image")
        
        contents = [image_prompt_in_english]
        if base_image_path:
            log_message(f"Carregando imagem base para edição: {base_image_path}", "Tool:generate_image")
            if not os.path.exists(base_image_path):
                return f"Erro: O arquivo da imagem base '{base_image_path}' não foi encontrado."
            image_part = Image.open(base_image_path)
            contents.append(image_part)
        
        image_model = genai.GenerativeModel(GEMINI_IMAGE_MODEL_NAME)
        response = image_model.generate_content(contents)
        
        image_part = next((part for part in response.candidates[0].content.parts if hasattr(part, 'blob') and part.blob.data), None)

        if not image_part:
             return "Erro: A API não retornou dados de imagem (blob) na resposta."

        image_bytes = image_part.blob.data
        
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

AGENT_TOOLS = [save_file, generate_image, create_cache]

# --- Funções de Comunicação e Arquivos ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")
def print_thought_message(message): print(f"\n🧠 [Pensamento do Agente]:\n{message}"); log_message(f"PENSAMENTO:\n{message}", "Agente")

def get_uploaded_files_info_from_user():
    """Função refatorada para usar a Files API através do módulo genai."""
    uploaded_file_objects = []
    uploaded_files_metadata = []
    
    api_files_list = []
    try:
        # Usa genai.list_files()
        api_files_list = list(genai.list_files())
    except Exception as e:
        log_message(f"Falha ao listar arquivos da API: {e}", "Sistema")
    
    if api_files_list:
        print_agent_message("Sistema", "Arquivos existentes na API:")
        for i, f in enumerate(api_files_list):
            print(f"  {i+1}. {f.display_name} (ID: {f.name})")
        
        if input("👤 Reutilizar arquivos existentes? (s/n) ➡️ ").lower() == 's':
            choices = input("👤 Números (ex: 1,3) ou 'todos': ➡️ ").lower()
            sel_indices = list(range(len(api_files_list))) if choices == 'todos' else [int(x.strip()) - 1 for x in choices.split(',') if x.strip().isdigit()]
            
            for idx in sel_indices:
                if 0 <= idx < len(api_files_list):
                    file_obj = api_files_list[idx]
                    uploaded_file_objects.append(file_obj)
                    uploaded_files_metadata.append({"file_id": file_obj.name, "display_name": file_obj.display_name, "mime_type": file_obj.mime_type})
                    print_agent_message("Sistema", f"✅ Arquivo '{file_obj.display_name}' selecionado para reutilização.")

    if input("👤 Fazer upload de NOVOS arquivos? (s/n) ➡️ ").lower() == 's':
        while True:
            fp_pattern = input("👤 Caminho/padrão do arquivo (ou 'fim'): ➡️ ").strip()
            if fp_pattern.lower() == 'fim': break
            
            expanded_paths = glob.glob(fp_pattern, recursive=True) if any(c in fp_pattern for c in ['*','?']) else ([fp_pattern] if os.path.exists(fp_pattern) else [])
            expanded_files = [f for f in expanded_paths if os.path.isfile(f)]

            if not expanded_files:
                print_agent_message("Sistema", f"❌ Nenhum arquivo encontrado para: '{fp_pattern}'"); continue
            
            for fp in expanded_files:
                dn = os.path.basename(fp)
                try:
                    print_agent_message("Sistema", f"Fazendo upload de '{dn}'...")
                    # Usa genai.upload_file()
                    file_obj = genai.upload_file(path=fp, display_name=dn)
                    uploaded_file_objects.append(file_obj)
                    uploaded_files_metadata.append({"file_id": file_obj.name, "display_name": dn, "mime_type": file_obj.mime_type, "user_path": fp})
                    print_agent_message("Sistema", f"✅ Novo arquivo '{dn}' (ID: {file_obj.name}) enviado.")
                    time.sleep(1)
                except Exception as e:
                    log_message(f"Erro no upload de '{dn}': {e}", "Sistema")
                    print_agent_message("Sistema", f"❌ Erro no upload de '{dn}': {e}")
    
    return uploaded_file_objects, uploaded_files_metadata


def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config_dict=None, system_instruction=None, tools=None):
    log_message(f"Iniciando chamada à API Gemini para {agent_name} (Modelo: {model_name})...", "Sistema")
    
    active_gen_config_dict = (gen_config_dict or {}).copy()
    
    thinking_config_data = active_gen_config_dict.pop("thinking_config", None)
    cached_content_name = active_gen_config_dict.pop("cached_content", None)
    
    thinking_config_obj = None
    if thinking_config_data and isinstance(thinking_config_data, dict):
        thinking_config_obj = genai.types.ThinkingConfig(**thinking_config_data)
        
    final_config_obj = genai.types.GenerationConfig(
        thinking_config=thinking_config_obj,
        cached_content=cached_content_name, 
        **active_gen_config_dict
    )
    
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
            "Execute o plano usando as ferramentas disponíveis (`save_file`, `generate_image`, `create_cache`). "
            "Ao final, forneça um resumo conciso do que foi feito."
        )
        log_message("Instância do Worker (v10.20) criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_context = f"Resultados de tarefas anteriores: {json.dumps(previous_results) if previous_results else 'Nenhum.'}"
        files_prompt_part = f"Arquivos de referência (use seus IDs para o cache): {[f'{meta['display_name']} (ID: {meta['file_id']})' for meta in uploaded_files_info]}"
        
        prompt_parts = [
            f"Contexto: {prompt_context}\n{files_prompt_part}",
            f"Objetivo Geral: {original_goal}",
            f"\nSua tarefa específica agora é: \"{sub_task_description}\"",
            "Execute a tarefa. Use as ferramentas disponíveis para pensar e agir."
        ]
        
        worker_gen_config = { 
            "thinking_config": {
                "thinking_budget": -1, 
                "include_thoughts": True
            } 
        } 

        last_result = previous_results[-1] if previous_results else {}
        cached_content_name = last_result.get(list(last_result.keys())[0], {}).get('text_content', '')
        if cached_content_name and cached_content_name.startswith('cachedContents/'):
            log_message(f"Usando cache da tarefa anterior: {cached_content_name}", "Worker")
            worker_gen_config["cached_content"] = cached_content_name
        elif self.task_manager.uploaded_file_objects:
             prompt_parts.extend(self.task_manager.uploaded_file_objects)


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
            "Você é um Gerenciador de Tarefas especialista. Sua função é decompor uma meta principal em um plano de sub-tarefas sequenciais. "
            "Se a meta envolver o uso de arquivos grandes ou repetitivos, o PRIMEIRO PASSO do plano deve ser criar um cache explícito usando a ferramenta 'create_cache'. "
            "Os passos seguintes devem então usar esse cache. "
            "Retorne o plano como um array JSON de strings, usando o esquema fornecido."
        )
        log_message("Instância do TaskManager (v10.20) criada.", "TaskManager")
        
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
                log_message(f"Erro ao decodificar JSON (mesmo com schema): {e}. Texto: '{json_text}'. Usando fallback.", "TaskManager")
        
        return [goal_to_decompose]
    
    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho (v10.20)...")
        self.current_task_list = self.decompose_goal(self.goal)
        
        if not self.current_task_list:
            print_agent_message("TaskManager", "Não foi possível decompor a meta. Encerrando."); return

        print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
        for i, task_desc in enumerate(self.current_task_list): print(f"  {i+1}. {task_desc}")
        
        if input("👤 Aprova este plano? (s/n) ➡️ ").strip().lower() != 's':
            print_agent_message("TaskManager", "Plano não aprovado. Encerrando."); return
        
        for task_description in self.current_task_list:
            task_result, _ = self.worker.execute_task(
                task_description, 
                self.executed_tasks_results, 
                self.uploaded_files_info, 
                self.goal
            )
            self.executed_tasks_results.append({task_description: task_result})

        print_agent_message("TaskManager", "Fluxo de trabalho concluído! Artefatos salvos em 'gemini_final_outputs'.")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v10.20 (Correção Client)"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
    
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()
    
    print_user_message("🎯 Defina a meta principal (digite 'FIM' em uma nova linha para concluir):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'FIM':
            break
        lines.append(line)
    initial_goal_input = "\n".join(lines)

    log_message(f"Meta recebida do usuário:\n---\n{initial_goal_input}\n---", "Usuário")
    
    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
    else:
        task_manager = TaskManager(initial_goal_input, uploaded_files, uploaded_files_meta)
        task_manager.run_workflow()

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"\n--- Execução ({SCRIPT_VERSION}) Finalizada ---")

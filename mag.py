import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import glob
from typing import List

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

# --- Modelos Gemini ---
# Usando modelos recentes que têm excelente suporte a 'tool use' e 'thinking'.
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-flash-preview-05-20" # O modelo Pro é ideal para 'thinking'
GEMINI_IMAGE_MODEL_NAME = "gemini-2.0-flash-preview-image-generation" 

# --- Funções de Utilidade ---
def sanitize_filename(name, allow_extension=True):
    """Limpa e sanitiza um nome de arquivo para evitar problemas de segurança e de sistema de arquivos."""
    if not name: return ""
    base_name, ext = os.path.splitext(name)
    base_name = re.sub(r'[^\w\s.-]', '', base_name).strip()
    base_name = re.sub(r'[-\s]+', '-', base_name)[:100]
    if not allow_extension or not ext: return base_name
    ext = "." + re.sub(r'[^\w-]', '', ext.lstrip('.')).strip()[:10]
    return base_name + ext

def log_message(message, source="Sistema"):
    """Escreve uma mensagem no arquivo de log com timestamp."""
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
@genai.tool
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

@genai.tool
def generate_image(image_prompt: str) -> str:
    """Gera uma imagem com base em uma descrição textual detalhada (prompt) e a salva como um arquivo PNG."""
    try:
        log_message(f"Iniciando geração de imagem com o prompt: '{image_prompt[:100]}...'", "Tool:generate_image")
        image_model = genai.GenerativeModel(GEMINI_IMAGE_MODEL_NAME)
        
        # A API para geração de imagem com o 1.5 Pro espera o prompt diretamente
        response = image_model.generate_content(
            f"Gere uma imagem com base na seguinte descrição: {image_prompt}",
        )
        
        image_part = next((part for part in response.candidates[0].content.parts if part.mime_type.startswith("image/")), None)
        if not image_part:
             return "Erro: A API não retornou dados de imagem na resposta. Verifique o prompt ou o modelo."

        image_bytes = image_part.blob.data
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fn_base = sanitize_filename(f"imagem_{image_prompt[:20]}_{ts}")
        filename = f"{fn_base}.png"
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        with open(full_path, "wb") as f: f.write(image_bytes)
            
        log_message(f"Imagem salva com sucesso como '{filename}'.", "Tool:generate_image")
        return f"Sucesso: Imagem gerada e salva como '{filename}'."
        
    except Exception as e:
        log_message(f"Erro na ferramenta generate_image: {e}\n{traceback.format_exc()}", "Tool:generate_image")
        return f"Erro ao gerar a imagem: {e}"

AGENT_TOOLS = [save_file, generate_image]

# --- Funções Auxiliares de Comunicação ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")

def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", model_name=GEMINI_TEXT_MODEL_NAME, gen_config_dict=None, system_instruction=None, tools=None):
    """Função de chamada à API Gemini, agora com suporte a 'thinking_config' dentro de gen_config_dict."""
    log_message(f"Iniciando chamada à API Gemini para {agent_name} (Modelo: {model_name})...", "Sistema")
    
    # --- MUDANÇA AQUI: Usa um dicionário para a configuração de geração ---
    active_gen_config = gen_config_dict or {}
    # Garante que os parâmetros padrão estejam presentes se não forem fornecidos
    active_gen_config.setdefault("temperature", 0.7)
    active_gen_config.setdefault("top_p", 0.95)
    active_gen_config.setdefault("top_k", 64)
    active_gen_config.setdefault("max_output_tokens", 8192)

    log_message(f"Usando generation_config: {active_gen_config}", "Sistema")
    
    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES} para {agent_name}...", "Sistema")
        try:
            content_parts = [genai.Part.from_text(str(p)) if not isinstance(p, genai.types.PartType) else p for p in prompt_parts]

            model_instance = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction, 
                tools=tools 
            )
            # A configuração é passada diretamente para generate_content
            response = model_instance.generate_content(
                content_parts,
                generation_config=active_gen_config 
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
            "Use as ferramentas `save_file` ou `generate_image` sempre que necessário para cumprir a tarefa. "
            "Pense passo a passo e use as ferramentas para alcançar o objetivo. "
            "Ao final, forneça um resumo conciso do que foi feito."
        )
        log_message("Instância do Worker (v10.2 com Thinking) criada.", "Worker")

    def execute_task(self, sub_task_description, previous_results, uploaded_files_info, original_goal):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")

        prompt_context = f"Resultados de tarefas anteriores: {json.dumps(previous_results) if previous_results else 'Nenhum.'}"
        files_prompt_part = f"Arquivos de referência: {json.dumps([f['display_name'] for f in uploaded_files_info]) if uploaded_files_info else 'Nenhum.'}"
        
        prompt_parts = [
            f"Contexto: {prompt_context}\n{files_prompt_part}",
            f"Objetivo Geral: {original_goal}",
            f"\nSua tarefa específica agora é: \"{sub_task_description}\"",
            "Execute a tarefa. Use as ferramentas disponíveis para pensar e agir."
        ]
        
        # --- MUDANÇA AQUI: Habilitando o Thinking para o Worker ---
        worker_gen_config = { "thinking_config": {"thinking_budget": 2048} }

        response = call_gemini_api_with_retry(
            prompt_parts,
            agent_display_name,
            model_name=self.model_name,
            system_instruction=self.system_instruction,
            tools=AGENT_TOOLS,
            gen_config_dict=worker_gen_config
        )

        if response is None:
            return {"text_content": "Falha: Sem resposta da API.", "saved_files": []}, []
        
        final_text_response = response.text if hasattr(response, 'text') else "Ação concluída através de ferramentas."
        log_message(f"Resposta final do Worker: {final_text_response[:500]}...", agent_display_name)
        return {"text_content": final_text_response, "saved_files": []}, []


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
            "Retorne o plano como um array JSON de strings. "
        )
        log_message("Instância do TaskManager (v10.2 com Thinking) criada.", "TaskManager")
        
    def decompose_goal(self, goal_to_decompose):
        agent_display_name = "Task Manager (Decomposição)"
        print_agent_message(agent_display_name, f"Decompondo meta: '{goal_to_decompose}'")

        prompt_parts = [ f"Meta Principal a ser decomposta: \"{goal_to_decompose}\"" ]
        
        # --- MUDANÇA AQUI: Habilitando o Thinking para o Planejamento ---
        planner_gen_config = { "thinking_config": {"thinking_budget": 1024} }

        response = call_gemini_api_with_retry(
            prompt_parts, 
            agent_display_name, 
            system_instruction=self.system_instruction,
            tools=AGENT_TOOLS, # Fornecer ferramentas ao planejador permite que ele verifique o estado antes de planejar
            gen_config_dict=planner_gen_config
        )
        
        if response and hasattr(response, 'text') and response.text:
            response_text = response.text
            log_message(f"Resposta da decomposição (bruta): {response_text}", agent_display_name)
            try:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
                json_str = match.group(1).strip() if match else response_text
                
                tasks = json.loads(json_str)
                if isinstance(tasks, list) and all(isinstance(task, str) for task in tasks):
                    return tasks
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                log_message(f"Erro ao processar JSON da decomposição: {e}. Usando fallback.", agent_display_name)
        
        return [goal_to_decompose]
    
    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho (v10.2)...")
        self.current_task_list = self.decompose_goal(self.goal)
        
        if not self.current_task_list:
            print_agent_message("TaskManager", "Não foi possível decompor a meta. Encerrando."); return

        print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
        for i, task_desc in enumerate(self.current_task_list): print(f"  {i+1}. {task_desc}")
        
        if input("👤 Aprova este plano? (s/n) ➡️ ").strip().lower() != 's':
            print_agent_message("TaskManager", "Plano não aprovado. Encerrando."); return
        
        for task_description in self.current_task_list:
            task_result, _ = self.worker.execute_task(
                task_description, self.executed_tasks_results, 
                self.uploaded_files_info, self.goal
            )
            self.executed_tasks_results.append({task_description: task_result})

        print_agent_message("TaskManager", "Fluxo de trabalho concluído! Artefatos salvos em 'gemini_final_outputs'.")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v10.2 (Thinking Config)"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
    
    print_user_message("🎯 Defina a meta principal (digite 'FIM' em uma nova linha para concluir):")
    lines = [line for line in iter(input, "FIM")]
    initial_goal_input = "\n".join(lines)

    log_message(f"Meta recebida do usuário:\n---\n{initial_goal_input}\n---", "Usuário")
    
    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
    else:
        task_manager = TaskManager(initial_goal_input)
        task_manager.run_workflow()

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"\n--- Execução ({SCRIPT_VERSION}) Finalizada ---")

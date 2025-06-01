import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import base64
import glob # Adicionado para encontrar o arquivo de cache mais recente

# --- Configuração dos Diretórios e Arquivos ---
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_agent_logs")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_final_outputs")
UPLOADED_FILES_CACHE_DIR = os.path.join(BASE_DIRECTORY, "gemini_uploaded_files_cache")

for directory in [LOG_DIRECTORY, OUTPUT_DIRECTORY, UPLOADED_FILES_CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

CURRENT_TIMESTAMP_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIRECTORY, f"agent_log_{CURRENT_TIMESTAMP_STR}.txt")
# O nome do arquivo de cache da sessão atual será definido em get_uploaded_files_info_from_user

# --- Constantes para Retentativas ---
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_FACTOR = 2

# --- Funções de Utilidade ---
def sanitize_filename(name):
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return name[:50]

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
GEMINI_IMAGE_GENERATION_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME} mais recente", "Sistema")
log_message(f"Modelo Gemini (geração de imagem via SDK): {GEMINI_IMAGE_GENERATION_MODEL_NAME}", "Sistema")

generation_config_text = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

generation_config_image = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "response_modalities": ['TEXT', 'IMAGE'],
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
    # ... (código idêntico à v8.1)
    log_message(f"Iniciando chamada à API Gemini para {agent_name}...", "Sistema")
    text_prompt_for_log = ""
    file_references_for_log = []

    active_gen_config = gen_config
    if active_gen_config is None:
        active_gen_config = generation_config_text if model_name == GEMINI_TEXT_MODEL_NAME else generation_config_image
        log_message(f"Nenhuma gen_config específica passada para {agent_name}, usando config padrão para modelo {model_name}.", "Sistema")

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
            
            if agent_name == "ImageWorker":
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
                if response.prompt_feedback.block_reason:
                    log_message(f"Bloqueio: {response.prompt_feedback.block_reason_message} ({response.prompt_feedback.block_reason})", agent_name)
            
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR; continue
            log_message(f"Falha após {MAX_API_RETRIES} tentativas (sem resposta utilizável para {agent_name}, Modelo: {model_name}).", agent_name)
            return None
        except Exception as e:
            log_message(f"Exceção na tentativa {attempt + 1}/{MAX_API_RETRIES} ({agent_name}, Modelo: {model_name}): {type(e).__name__} - {e}", agent_name)
            log_message(f"Traceback: {traceback.format_exc()}", agent_name)
            if isinstance(e, genai.types.BlockedPromptException): log_message(f"Exceção Prompt Bloqueado: {e}", agent_name)
            elif isinstance(e, genai.types.StopCandidateException): log_message(f"Exceção Parada de Candidato: {e}", agent_name)
            if attempt < MAX_API_RETRIES - 1:
                log_message(f"Aguardando {current_retry_delay}s...", "Sistema"); time.sleep(current_retry_delay); current_retry_delay *= RETRY_BACKOFF_FACTOR
            else:
                log_message(f"Máximo de {MAX_API_RETRIES} tentativas. Falha API Gemini ({agent_name}, Modelo: {model_name}).", agent_name)
                return None
    log_message(f"call_gemini_api_with_retry ({agent_name}, Modelo: {model_name}) terminou sem retorno explícito após loop.", "Sistema")
    return None

# --- Funções de Arquivos ---
def get_most_recent_cache_file():
    """Encontra o arquivo de cache de metadados de upload mais recente."""
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
    """Carrega metadados de um arquivo de cache JSON."""
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

def get_uploaded_files_info_from_user():
    uploaded_file_objects = [] 
    uploaded_files_metadata = [] 
    reused_file_ids = set() # Para rastrear IDs de arquivos reutilizados

    print_agent_message("Sistema", "Verificando arquivos existentes na API Gemini...")
    api_files_list = list(genai.list_files()) # Converte para lista para evitar problemas com iterador
    api_files_dict = {f.name: f for f in api_files_list}
    log_message(f"Encontrados {len(api_files_dict)} arquivos na API.", "Sistema")

    most_recent_cache_path = get_most_recent_cache_file()
    cached_metadata_list = []
    if most_recent_cache_path:
        log_message(f"Tentando carregar cache de metadados de: {most_recent_cache_path}", "Sistema")
        cached_metadata_list = load_cached_files_metadata(most_recent_cache_path)
    
    if cached_metadata_list and api_files_dict:
        reusable_files_from_cache = []
        print_agent_message("Sistema", "Arquivos de sessões anteriores encontrados na API e no cache local:")
        for idx, cached_file_meta in enumerate(cached_metadata_list):
            file_id = cached_file_meta.get("file_id")
            display_name = cached_file_meta.get("display_name", "Nome Desconhecido")
            if file_id in api_files_dict: # Verifica se o arquivo do cache ainda existe na API
                reusable_files_from_cache.append(cached_file_meta)
                print(f"  {len(reusable_files_from_cache)}. {display_name} (ID: {file_id}, Tipo: {cached_file_meta.get('mime_type')})")
        
        if reusable_files_from_cache:
            print_user_message("Deseja reutilizar algum desses arquivos? (s/n)")
            if input("➡️ ").strip().lower() == 's':
                print_user_message("Digite os números dos arquivos para reutilizar, separados por vírgula (ex: 1,3). Ou 'todos':")
                choices_str = input("➡️ ").strip().lower()
                selected_indices_to_try = []

                if choices_str == 'todos':
                    selected_indices_to_try = list(range(len(reusable_files_from_cache)))
                else:
                    try:
                        selected_indices_to_try = [int(x.strip()) - 1 for x in choices_str.split(',')]
                    except ValueError:
                        print("❌ Entrada inválida para seleção de arquivos.")
                        log_message("Entrada inválida do usuário para seleção de arquivos cacheados.", "Sistema")
                
                for selected_idx in selected_indices_to_try:
                    if 0 <= selected_idx < len(reusable_files_from_cache):
                        chosen_meta = reusable_files_from_cache[selected_idx]
                        file_id_to_reuse = chosen_meta["file_id"]
                        if file_id_to_reuse in reused_file_ids:
                            print(f"ℹ️ Arquivo '{chosen_meta['display_name']}' já selecionado para reutilização.")
                            continue
                        try:
                            print_agent_message("Sistema", f"Obtendo arquivo '{chosen_meta['display_name']}' (ID: {file_id_to_reuse}) da API...")
                            file_obj = genai.get_file(name=file_id_to_reuse) 
                            uploaded_file_objects.append(file_obj)
                            uploaded_files_metadata.append(chosen_meta) 
                            reused_file_ids.add(file_id_to_reuse)
                            print(f"✅ Arquivo '{file_obj.display_name}' reutilizado.")
                            log_message(f"Arquivo '{file_obj.display_name}' (ID: {file_id_to_reuse}) reutilizado da API.", "Sistema")
                        except Exception as e:
                            print(f"❌ Erro ao obter arquivo '{chosen_meta['display_name']}' da API: {e}")
                            log_message(f"Erro ao obter arquivo '{file_id_to_reuse}' da API: {e}", "Sistema")
                    else:
                        print(f"❌ Índice inválido: {selected_idx + 1}")
        else:
            print_agent_message("Sistema", "Nenhum arquivo do cache local foi encontrado ativo na API para reutilização ou o cache está vazio.")

    print_user_message("Adicionar NOVOS arquivos complementares (além dos reutilizados)? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        print_agent_message("Sistema", "Fazendo upload de novos arquivos...")
        while True:
            print_user_message("Caminho do novo arquivo (ou 'fim'):")
            fp = input("➡️ ").strip()
            if fp.lower() == 'fim': break
            if not os.path.exists(fp) or not os.path.isfile(fp):
                print(f"❌ Arquivo '{fp}' inválido."); log_message(f"Arquivo '{fp}' inválido.", "Sistema"); continue
            
            dn = os.path.basename(fp)
            # Verifica se um arquivo com o mesmo nome de exibição já foi reutilizado (para evitar confusão)
            if any(meta.get("display_name") == dn for meta in uploaded_files_metadata):
                print_user_message(f"⚠️ Um arquivo chamado '{dn}' já foi reutilizado ou adicionado. Continuar com novo upload? (s/n)")
                if input("➡️ ").strip().lower() != 's':
                    continue
            try:
                print_agent_message("Sistema", f"Upload de '{dn}'...")
                # Tenta especificar o mime_type se for um tipo comum não detectado
                mime_type_upload = None
                if dn.endswith(".md"): mime_type_upload = "text/markdown"
                elif dn.endswith(".py"): mime_type_upload = "text/x-python"
                elif dn.endswith(".cpp"): mime_type_upload = "text/x-c++src" # Exemplo
                elif dn.endswith(".h"): mime_type_upload = "text/x-chdr" # Exemplo

                uf_args = {'path': fp, 'display_name': dn}
                if mime_type_upload:
                    uf_args['mime_type'] = mime_type_upload
                
                uf = genai.upload_file(**uf_args)

                uploaded_file_objects.append(uf)
                fm = {"user_path":fp,"display_name":uf.display_name,"file_id":uf.name,"uri":uf.uri,"mime_type":uf.mime_type,"size_bytes":uf.size_bytes,"state":str(uf.state)}
                uploaded_files_metadata.append(fm)
                reused_file_ids.add(uf.name) # Adiciona à lista para evitar duplicação se o cache for usado na mesma sessão
                print(f"✅ '{dn}' (ID: {uf.name}, Tipo: {uf.mime_type}) enviado!")
                log_message(f"Novo arquivo '{dn}' (ID: {uf.name}, URI: {uf.uri}, Tipo: {uf.mime_type}, Tamanho: {uf.size_bytes}B) enviado.", "Sistema")
            except Exception as e:
                print(f"❌ Erro no upload de '{fp}': {e}")
                log_message(f"Erro no upload de '{fp}': {e}\n{traceback.format_exc()}", "Sistema")

    if uploaded_files_metadata:
        try:
            current_session_cache_path = os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json")
            with open(current_session_cache_path, "w", encoding="utf-8") as f:
                json.dump(uploaded_files_metadata, f, indent=4)
            log_message(f"Metadados dos arquivos da sessão atual ({len(uploaded_files_metadata)} arquivos) salvos em: {current_session_cache_path}", "Sistema")
        except Exception as e:
            log_message(f"Erro ao salvar metadados dos uploads da sessão atual: {e}", "Sistema")
            
    return uploaded_file_objects, uploaded_files_metadata

def format_uploaded_files_info_for_prompt_text(files_metadata_list):
    # ... (código idêntico à v8.1)
    if not files_metadata_list: return "Nenhum arquivo complementar fornecido."
    txt = "Arquivos complementares carregados (referencie pelo 'Nome de Exibição' ou 'ID do Arquivo'):\n"
    for m in files_metadata_list: txt += f"- Nome: {m['display_name']} (ID: {m['file_id']}, Tipo: {m['mime_type']})\n"
    return txt

# --- Classe ImageWorker ---
class ImageWorker:
    # ... (código idêntico à v8.1)
    def __init__(self):
        self.model_name = GEMINI_IMAGE_GENERATION_MODEL_NAME
        self.generation_config = generation_config_image
        log_message(f"Instância do ImageWorker criada para o modelo Gemini: {self.model_name}", "ImageWorker")
        log_message(f"ImageWorker usará generation_config: {self.generation_config}", "ImageWorker")

    def generate_image(self, text_prompt_for_image):
        agent_display_name = "ImageWorker"
        print_agent_message(agent_display_name, f"Solicitando geração de imagem com prompt: '{text_prompt_for_image[:100]}...'")
        
        generation_instruction_prompt = (
            f"Gere uma imagem de alta qualidade que represente o seguinte conceito ou descrição detalhada:\n\n"
            f"\"{text_prompt_for_image}\"\n\n"
            f"A imagem deve ser retornada diretamente como dados inline. "
            f"Você também pode fornecer uma breve descrição textual ou título para a imagem gerada, se desejar."
        )
        
        log_message(f"Prompt construído para {self.model_name} (geração de imagem):\n{generation_instruction_prompt}", agent_display_name)

        response_object = call_gemini_api_with_retry(
            prompt_parts=[generation_instruction_prompt],
            agent_name=agent_display_name,
            model_name=self.model_name,
            gen_config=self.generation_config
        )

        if response_object is None:
            log_message(f"Falha na chamada à API para {self.model_name} no ImageWorker (retornou None).", agent_display_name)
            return "Falha na geração da imagem (API não respondeu)."
        
        image_base64_str = None
        returned_text_content = ""

        if response_object.candidates and response_object.candidates[0].content and response_object.candidates[0].content.parts:
            parts = response_object.candidates[0].content.parts
            log_message(f"ImageWorker: Processando {len(parts)} partes da resposta.", agent_display_name)

            for i, part in enumerate(parts):
                log_message(f"ImageWorker: Analisando parte {i}: {str(part)[:200]}...", agent_display_name)
                if part.text is not None and part.text.strip():
                    current_part_text = part.text.strip()
                    returned_text_content += (current_part_text + "\n") if current_part_text else ""
                    log_message(f"ImageWorker: Texto encontrado na parte {i}: '{current_part_text[:100]}...'", agent_display_name)
                elif part.inline_data and part.inline_data.data:
                    mime_type = part.inline_data.mime_type
                    image_bytes = part.inline_data.data
                    
                    if mime_type.startswith("image/"):
                        image_base64_str = base64.b64encode(image_bytes).decode('utf-8')
                        log_message(f"Sucesso! Imagem (Tipo: {mime_type}) convertida para string base64, recebida de {self.model_name} na parte {i}.", agent_display_name)
                    else:
                        log_message(f"Alerta: Mime type retornado ({mime_type}) na parte {i} não é de imagem, mas inline_data foi encontrado.", agent_display_name)
            
            returned_text_content = returned_text_content.strip()

            if image_base64_str:
                return image_base64_str
            else:
                log_message(f"API Gemini (Modelo: {self.model_name}) retornou 'parts', mas nenhuma continha 'inline_data' de imagem.", agent_display_name)
                if response_object.prompt_feedback and response_object.prompt_feedback.block_reason:
                     log_message(f"Geração de imagem bloqueada: {response_object.prompt_feedback.block_reason_message}", agent_display_name)
                     return f"Falha na geração da imagem: Bloqueado ({response_object.prompt_feedback.block_reason_message})"
                if returned_text_content:
                    return f"Falha na geração da imagem: Nenhuma imagem encontrada, mas o modelo respondeu com texto: '{returned_text_content[:200]}...'"
                return "Falha na geração da imagem: Nenhuma imagem encontrada nas partes da resposta."

        elif response_object.prompt_feedback and response_object.prompt_feedback.block_reason:
            log_message(f"Geração de imagem bloqueada (sem 'candidates' ou 'parts' utilizáveis): {response_object.prompt_feedback.block_reason_message}", agent_display_name)
            return f"Falha na geração da imagem: Bloqueado ({response_object.prompt_feedback.block_reason_message})"
        
        log_message(f"Falha na geração da imagem com {self.model_name}. Nenhuma 'candidates' ou 'parts' utilizáveis na resposta. Resposta: {response_object}", agent_display_name)
        return "Falha na geração da imagem (resposta da API inesperada ou vazia)."


# --- Classe TaskManager ---
class TaskManager:
    # ... (código de __init__, decompose_task, confirm_new_tasks_with_llm, evaluate_and_select_image_concepts, run_workflow, extract_structured_output, validate_and_save_final_output idêntico à v8.2, pois as correções para o bug do dict já estavam na v8.1 e foram mantidas na v8.2)
    def __init__(self):
        self.gemini_text_model_name = GEMINI_TEXT_MODEL_NAME
        self.worker = Worker()
        self.image_worker = ImageWorker()
        self.task_list = []
        self.completed_tasks_results = []
        log_message("Instância do TaskManager criada.", "TaskManager")

    def decompose_task(self, main_goal, uploaded_file_objects, files_metadata_for_prompt_text):
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
3.  (Opcional) "Avaliar a imagem gerada." (Se a avaliação for parte do fluxo)

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
                        # Se não for lista de strings nem lista de dicts com 'tarefa', tenta tratar como lista de strings (caso o LLM retorne strings simples sem aspas corretas no JSON)
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
                    # Tenta interpretar como uma lista de strings simples se não houver JSON
                    lines = [line.strip().replace('"', '').replace(',', '') for line in response_text.splitlines() if line.strip() and not line.strip().startswith(('[', ']'))]
                    if lines:
                        self.task_list = lines
                        log_message(f"Decomposição interpretada como lista de strings simples: {self.task_list}", agent_display_name)
                        print_agent_message(agent_display_name, f"Tarefas decompostas (interpretadas): {self.task_list}")
                        return True
                    print_agent_message(agent_display_name, f"Decomposição não retornou JSON no formato esperado.")


            except json.JSONDecodeError as e:
                # Se o JSON falhar, tenta tratar a resposta como uma lista de strings separadas por nova linha
                log_message(f"Falha ao decodificar JSON da decomposição: {e}. Tentando interpretar como lista de strings.", agent_display_name)
                lines = [line.strip().replace('"', '').replace(',', '') for line in response_text.splitlines() if line.strip() and not line.strip().startswith(('[', ']'))]
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
                        if isinstance(item, str):
                            approved_tasks_final.append(item)
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str):
                            approved_tasks_final.append(item["tarefa"])
                        else:
                            log_message(f"Item de tarefa aprovada ignorado (formato inesperado): {item}", agent_name)
                else:
                     log_message(f"Resposta de aprovação não é uma lista: {parsed_response}", agent_name)

                print_agent_message(agent_name, f"Novas tarefas aprovadas (strings): {approved_tasks_final}")
            except Exception as ex:
                log_message(f"Erro ao decodificar/processar aprovação: {ex}. Resp: {response}", agent_name)
                log_message(f"Traceback: {traceback.format_exc()}", agent_name)
        else:
            log_message("Falha API na validação de novas tarefas.", agent_name)
        return approved_tasks_final

    def evaluate_and_select_image_concepts(self, original_goal, image_task_results, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_display_name = "Task Manager (Avaliação de Conceitos de Imagem)"
        print_agent_message(agent_display_name, "Avaliando conceitos de imagem gerados/tentados...")

        summary_of_image_attempts = "Resumo das tentativas de geração de imagem:\n"
        if not image_task_results:
            summary_of_image_attempts += "Nenhuma tentativa de geração de imagem foi registrada.\n"
        for i, res in enumerate(image_task_results):
            summary_of_image_attempts += f"Tentativa {i+1}:\n"
            summary_of_image_attempts += f"  - Prompt Usado: {res.get('image_prompt_used', 'N/A')}\n"
            is_base64_success = isinstance(res.get("result"), str) and len(res.get("result", "")) > 100 and not str(res.get("result", "")).startswith("Falha")
            summary_of_image_attempts += f"  - Geração Bem-Sucedida: {'Sim' if is_base64_success else 'Não'}\n"
            if not is_base64_success:
                 summary_of_image_attempts += f"  - Resultado/Erro: {str(res.get('result'))[:200]}...\n"
            summary_of_image_attempts += "\n"

        prompt_text_part = f"""
Você é um Diretor de Arte especialista. Seu objetivo é analisar os resultados das tentativas de geração de imagem para a meta: "{original_goal}".
Considere também os arquivos complementares: {files_metadata_for_prompt_text}

Abaixo estão os resumos das tentativas de geração de imagem. Você NÃO PODE VER AS IMAGENS, apenas os prompts usados e se a geração foi bem-sucedida.
{summary_of_image_attempts}

Com base APENAS nos prompts usados e no sucesso/falha da geração, identifique TODOS OS PROMPTS DE IMAGEM que você considera válidos e que atendem aos critérios do objetivo original (ex: "arte conceitual de Eirenia").
Se múltiplas imagens foram geradas com sucesso e são válidas, inclua todos os prompts correspondentes.
Se algumas falharam mas o prompt era bom, você pode incluí-lo para uma nova tentativa.
Se nenhuma tentativa foi feita ou nenhum prompt parece adequado, retorne uma lista vazia.

Retorne sua resposta como um array JSON de strings, onde cada string é um prompt de imagem considerado válido.
Exemplo: ["prompt para imagem 1 válida", "prompt para imagem 2 válida", "prompt para imagem 3 que falhou mas era bom"]
Se nenhum, retorne: []

Prompts Selecionados (JSON Array):
"""
        prompt_parts_for_api = [prompt_text_part] + uploaded_file_objects
        llm_response_text = call_gemini_api_with_retry(
            prompt_parts_for_api,
            agent_display_name,
            model_name=self.gemini_text_model_name,
            gen_config=generation_config_text
        )
        
        selected_prompts_strings = []
        if llm_response_text:
            try:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*?\])', llm_response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    parsed_list = json.loads(json_str)
                    if isinstance(parsed_list, list) and all(isinstance(p, str) for p in parsed_list):
                        selected_prompts_strings = parsed_list
                        log_message(f"LLM selecionou {len(selected_prompts_strings)} prompts válidos.", agent_display_name)
                    else:
                        log_message(f"Resposta do LLM para seleção de prompts não é uma lista de strings: {parsed_list}", agent_display_name)
                else:
                    log_message(f"Não foi possível extrair JSON da resposta do LLM para seleção de prompts: {llm_response_text}", agent_display_name)
            except json.JSONDecodeError as e:
                log_message(f"Erro ao decodificar JSON da seleção de prompts: {e}. Resposta: {llm_response_text}", agent_display_name)
        
        validated_concepts = []
        if selected_prompts_strings:
            print_agent_message(agent_display_name, f"LLM considerou {len(selected_prompts_strings)} prompts como válidos.")
            for sel_prompt in selected_prompts_strings:
                found_concept = False
                for res_attempt in image_task_results:
                    if res_attempt.get('image_prompt_used', '').strip() == sel_prompt.strip():
                        validated_concepts.append({
                            "image_prompt_selected": sel_prompt.strip(),
                            "result": res_attempt.get("result")
                        })
                        found_concept = True
                        break
                if not found_concept:
                    validated_concepts.append({
                        "image_prompt_selected": sel_prompt.strip(),
                        "result": "Nova tentativa ou geração necessária para este prompt validado."
                    })
                    log_message(f"Prompt '{sel_prompt.strip()}' validado pelo LLM, mas sem resultado de imagem correspondente nas tentativas atuais.", agent_display_name)
        
        if not validated_concepts and image_task_results:
            log_message("LLM não retornou prompts válidos, usando o primeiro resultado de tentativa como fallback se houver.", agent_display_name)
            validated_concepts.append({
                "image_prompt_selected": image_task_results[0].get('image_prompt_used'),
                "result": image_task_results[0].get("result")
            })
        elif not validated_concepts:
            log_message("Nenhuma tentativa de imagem e nenhum prompt validado pelo LLM.", agent_display_name)
            return []

        log_message(f"Conceitos validados para prosseguir: {len(validated_concepts)}", agent_display_name)
        return validated_concepts


    def run_workflow(self, initial_goal, uploaded_file_objects, uploaded_files_metadata):
        agent_display_name = "Task Manager"
        print_agent_message(agent_display_name, "Iniciando fluxo de trabalho...")
        log_message(f"Meta inicial: {initial_goal}", agent_display_name)
        
        files_metadata_for_prompt_text = format_uploaded_files_info_for_prompt_text(uploaded_files_metadata)

        if not self.decompose_task(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text):
            print_agent_message(agent_display_name, "Falha na decomposição da tarefa. Encerrando.")
            return
        if not self.task_list:
            print_agent_message(agent_display_name, "Nenhuma tarefa decomposta. Encerrando.")
            return
        
        print_agent_message(agent_display_name, "--- PLANO DE TAREFAS INICIAL ---")
        for i, task_item in enumerate(self.task_list): print(f"  {i+1}. {task_item}")
        print_user_message("Aprova este plano? (s/n)"); user_approval = input("➡️ ").strip().lower()
        log_message(f"Usuário {('aprovou' if user_approval == 's' else 'rejeitou')} o plano.", "UsuárioInput")
        if user_approval != 's': print_agent_message(agent_display_name, "Plano rejeitado. Encerrando."); return

        image_generation_attempts = []

        current_task_index = 0
        while current_task_index < len(self.task_list):
            current_task_description = self.task_list[current_task_index] 
            if isinstance(current_task_description, dict) and "tarefa" in current_task_description:
                current_task_description = current_task_description["tarefa"]
            elif not isinstance(current_task_description, str):
                log_message(f"Item de tarefa inválido encontrado: {current_task_description}. Pulando.", agent_display_name)
                current_task_index += 1
                continue

            print_agent_message(agent_display_name, f"Próxima tarefa ({current_task_index + 1}/{len(self.task_list)}): {current_task_description}")

            task_result_for_completed_list = None
            suggested_new_tasks_raw = []
            
            if current_task_description.startswith("TASK_GERAR_IMAGEM:"):
                image_prompt_description = "N/A - Descrição não encontrada"
                if self.completed_tasks_results and self.completed_tasks_results[-1]["result"]:
                    prev_result = self.completed_tasks_results[-1]["result"]
                    if isinstance(prev_result, str):
                        image_prompt_description = prev_result.strip()
                    else:
                        log_message(f"Resultado da tarefa anterior ({self.completed_tasks_results[-1]['task']}) não é string: {type(prev_result)}", agent_display_name)
                
                if not image_prompt_description or image_prompt_description == "N/A - Descrição não encontrada":
                    task_result_for_completed_list = "Erro: Descrição da imagem vazia ou não encontrada na tarefa anterior."
                    log_message(task_result_for_completed_list, agent_display_name)
                else:
                    print_agent_message(agent_display_name, f"Delegando para ImageWorker com prompt: '{image_prompt_description[:70]}...'")
                    task_result_for_completed_list = self.image_worker.generate_image(image_prompt_description)
                
                image_generation_attempts.append({
                    "image_prompt_used": image_prompt_description,
                    "result": task_result_for_completed_list 
                })
                self.completed_tasks_results.append({"task": current_task_description, "result": task_result_for_completed_list})
            
            elif current_task_description.startswith("TASK_AVALIAR_IMAGENS:"):
                validated_image_concepts = self.evaluate_and_select_image_concepts(
                    initial_goal, image_generation_attempts, uploaded_file_objects, files_metadata_for_prompt_text
                )
                self.completed_tasks_results.append({
                    "task": current_task_description,
                    "result": validated_image_concepts, 
                })
                log_message(f"Tarefa '{current_task_description}' concluída. {len(validated_image_concepts)} conceito(s) de imagem validado(s).", agent_display_name)
                current_task_index += 1; time.sleep(1); continue

            else: 
                context_summary = "Resultados anteriores:\n" + ("Nenhum.\n" if not self.completed_tasks_results else "".join([f"- '{r['task']}': {str(r.get('result','N/A'))[:200]}...\n" for r in self.completed_tasks_results]))
                context_summary += f"\nArquivos: {files_metadata_for_prompt_text}\nObjetivo: {initial_goal}\n"
                log_message(f"Enviando '{current_task_description}' para Worker.", agent_display_name)
                task_result_text, suggested_new_tasks_raw = self.worker.execute_sub_task(current_task_description, context_summary, uploaded_file_objects)
                
                task_result_for_completed_list = task_result_text
                if task_result_text is None:
                    task_result_for_completed_list = "Falha crítica na execução da tarefa pelo Worker (retornou None)."
                self.completed_tasks_results.append({"task": current_task_description, "result": task_result_for_completed_list})

            if not current_task_description.startswith("TASK_AVALIAR_IMAGENS:"):
                log_message(f"Resultado da tarefa '{current_task_description}': {str(task_result_for_completed_list)[:200]}...", agent_display_name)
                log_message(f"Tarefa '{current_task_description}' concluída.", agent_display_name)

            if suggested_new_tasks_raw:
                print_agent_message(agent_display_name, f"Worker sugeriu: {suggested_new_tasks_raw}")
                approved_tasks_strings = self.confirm_new_tasks_with_llm(initial_goal, self.task_list, suggested_new_tasks_raw, uploaded_file_objects, files_metadata_for_prompt_text)
                if approved_tasks_strings:
                    for nt_idx, nt_string in enumerate(approved_tasks_strings):
                        if nt_string not in self.task_list:
                            self.task_list.insert(current_task_index + 1 + nt_idx, nt_string)
                            log_message(f"Nova tarefa APROVADA '{nt_string}' inserida na lista.", agent_display_name)
                    print_agent_message(agent_display_name, f"Lista de tarefas atualizada: {[str(t)[:100]+'...' for t in self.task_list]}")
                else: print_agent_message(agent_display_name, "Nenhuma nova tarefa sugerida aprovada.")
            
            current_task_index += 1; time.sleep(1) 

        print_agent_message(agent_display_name, "Todas as tarefas processadas.")
        self.validate_and_save_final_output(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text)

    def extract_structured_output(self, llm_response_text):
        output_type, main_content, evaluation_text = "TEXTO_GERAL", llm_response_text, llm_response_text
        if not llm_response_text:
            log_message("extract_structured_output recebeu resposta vazia do LLM.", "TM(Val)")
            return "TEXTO_GERAL", "Erro: Resposta vazia do LLM de validação.", "Erro: Resposta vazia do LLM de validação."

        type_match = re.search(r"TIPO_DE_SAIDA_PRINCIPAL:\s*([\w_]+)", llm_response_text, re.IGNORECASE)
        if type_match: output_type = type_match.group(1).upper()
        
        content_match = re.search(r"CONTEUDO_PRINCIPAL_PARA_SALVAR:\s*([\s\S]*?)(?=AVALIACAO_GERAL:|$)", llm_response_text, re.IGNORECASE | re.DOTALL)
        if content_match:
            main_content = content_match.group(1).strip()
            main_content = re.sub(r'^```[a-zA-Z]*\s*\n|```\s*$', '', main_content, flags=re.DOTALL).strip()

        eval_match = re.search(r"AVALIACAO_GERAL:\s*([\s\S]*)", llm_response_text, re.IGNORECASE | re.DOTALL)
        if eval_match: evaluation_text = eval_match.group(1).strip()

        if main_content == llm_response_text and evaluation_text != llm_response_text:
            temp_content = llm_response_text
            if type_match: temp_content = temp_content.replace(type_match.group(0), "", 1)
            if eval_match: temp_content = temp_content.replace(eval_match.group(0), "", 1)
            temp_content = re.sub(r"CONTEUDO_PRINCIPAL_PARA_SALVAR:\s*", "", temp_content, flags=re.IGNORECASE).strip()
            main_content = temp_content if temp_content else main_content

        log_message(f"Output Extraído: Tipo={output_type}, Conteúdo~{main_content[:100]}..., Avaliação~{evaluation_text[:100]}...", "TM(Val)")
        return output_type, main_content, evaluation_text


    def validate_and_save_final_output(self, original_goal, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_display_name = "Task Manager (Validação)"
        print_agent_message(agent_display_name, "Validando resultado final...")
        if not self.completed_tasks_results:
            print_agent_message(agent_display_name, "Nenhuma tarefa concluída. Nada para validar ou salvar.")
            return

        results_summary_text = f"Meta Original: {original_goal}\nArquivos: {files_metadata_for_prompt_text}\nResultados Sub-tarefas:\n"
        
        validated_and_generated_images = [] # Lista de dicts: {"prompt": str, "base64": str}
        
        evaluation_task_result_obj = next((res for res in self.completed_tasks_results if res["task"].startswith("TASK_AVALIAR_IMAGENS:")), None)

        if evaluation_task_result_obj and isinstance(evaluation_task_result_obj.get("result"), list):
            validated_concepts_list = evaluation_task_result_obj.get("result")
            results_summary_text += f"- Tarefa: {evaluation_task_result_obj['task']}\n  Conceitos Avaliados ({len(validated_concepts_list)} validados pelo Diretor de Arte):\n"
            for idx, concept in enumerate(validated_concepts_list):
                prompt = concept.get("image_prompt_selected", "N/A")
                gen_result = concept.get("result")
                is_success = isinstance(gen_result, str) and len(gen_result) > 100 and not gen_result.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', gen_result.strip())
                
                status_text = "Sucesso (imagem base64)" if is_success else str(gen_result)
                results_summary_text += f"    - Conceito {idx+1} (Prompt: {prompt[:100]}...): {status_text[:100]}...\n"
                if is_success:
                    validated_and_generated_images.append({"prompt": prompt, "base64": gen_result.strip()})
            results_summary_text += "\n"
        else:
            log_message("Nenhum resultado da tarefa TASK_AVALIAR_IMAGENS encontrado ou não é uma lista.", agent_display_name)

        # Adiciona outros resultados ao sumário
        for item in self.completed_tasks_results:
            if not item["task"].startswith("TASK_AVALIAR_IMAGENS:"):
                result_display = item.get('result', 'N/A')
                if item['task'].startswith("TASK_GERAR_IMAGEM:"):
                    is_base64_like = isinstance(result_display,str) and len(result_display)>100 and not result_display.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', result_display.strip())
                    if is_base64_like:
                        was_validated = any(img_info["base64"] == result_display.strip() for img_info in validated_and_generated_images)
                        result_display = f"[IMAGEM GERADA - {'Validada e salva' if was_validated else 'Não validada / Outra variação'}]"
                    else:
                        result_display = f"[TENTATIVA DE GERAR IMAGEM - Falhou: {result_display}]"
                results_summary_text += f"- Tarefa: {item['task']}\n  Resultado: {str(result_display)[:300]}...\n\n"

        prompt_text_part_validation = f"""
Você é um Gerenciador de Tarefas especialista em validação. Meta original: "{original_goal}"
Arquivos: {files_metadata_for_prompt_text}
Resultados das sub-tarefas:
{results_summary_text}
Com base nisso, sua tarefa é:
1.  Identificar o TIPO DE SAÍDA PRINCIPAL. Se uma ou mais imagens foram validadas e geradas com sucesso, use IMAGEM_PNG_BASE64. Caso contrário, TEXTO_GERAL.
2.  Fornecer o CONTEÚDO PRINCIPAL PARA SALVAR (para o arquivo de avaliação).
    - Se o tipo for IMAGEM_PNG_BASE64, use a placeholder "[IMAGEM_BASE64_AQUI_PRINCIPAL]" (referindo-se à primeira imagem validada).
    - Para TEXTO_GERAL, forneça um resumo textual dos resultados.
3.  Fornecer uma AVALIAÇÃO GERAL da execução, mencionando quantas imagens foram validadas e efetivamente salvas (se houver).
Formato:
TIPO_DE_SAIDA_PRINCIPAL: [TIPO]
CONTEUDO_PRINCIPAL_PARA_SALVAR:
[Conteúdo textual ou "[IMAGEM_BASE64_AQUI_PRINCIPAL]"]
AVALIACAO_GERAL:
[Avaliação]
Siga estritamente.
"""
        llm_full_response = call_gemini_api_with_retry(
            [prompt_text_part_validation] + uploaded_file_objects,
            agent_display_name,
            model_name=self.gemini_text_model_name,
            gen_config=generation_config_text
        )

        if llm_full_response:
            print_agent_message(agent_display_name, f"--- RESPOSTA VALIDAÇÃO (BRUTA) ---\n{llm_full_response}")
            output_type, main_content_from_llm, evaluation_text = self.extract_structured_output(llm_full_response)
            
            actual_main_content_to_save_for_assessment = main_content_from_llm

            if output_type == "IMAGEM_PNG_BASE64":
                if not validated_and_generated_images: # LLM indicou imagem, mas não temos nenhuma.
                    output_type = "TEXTO_GERAL"
                    actual_main_content_to_save_for_assessment = f"Erro: LLM indicou saída de imagem, mas nenhuma imagem válida foi gerada/selecionada.\nConteúdo original do LLM: {main_content_from_llm if main_content_from_llm != '[IMAGEM_BASE64_AQUI_PRINCIPAL]' else ''}"
                    evaluation_text += "\nNOTA: O tipo de saída foi alterado para TEXTO_GERAL pois nenhuma imagem válida estava disponível."
                    log_message("LLM indicou IMAGEM_PNG_BASE64, mas nenhuma imagem válida disponível. Revertendo para TEXTO_GERAL.", agent_display_name)
                elif main_content_from_llm == "[IMAGEM_BASE64_AQUI_PRINCIPAL]": # Usa a primeira imagem para o snippet de avaliação
                     actual_main_content_to_save_for_assessment = validated_and_generated_images[0]["base64"]

            print_agent_message(agent_display_name, f"Tipo Saída Final (para avaliação): {output_type}")
            print_agent_message(agent_display_name, f"Conteúdo Principal (para avaliação): {str(actual_main_content_to_save_for_assessment)[:200]}...")
            print_agent_message(agent_display_name, f"Avaliação Geral: {evaluation_text}")

            goal_slug = sanitize_filename(original_goal)
            assessment_file_name = os.path.join(OUTPUT_DIRECTORY, f"avaliacao_completa_{goal_slug}_{CURRENT_TIMESTAMP_STR}.txt")
            try:
                with open(assessment_file_name, "w", encoding="utf-8") as f:
                    f.write(f"Meta: {original_goal}\nArquivos: {files_metadata_for_prompt_text}\n--- RESULTADOS SUB-TAREFAS ---\n")
                    for item in self.completed_tasks_results:
                        res_disp = item.get('result', 'N/A')
                        if item['task'].startswith("TASK_AVALIAR_IMAGENS:") and isinstance(res_disp, list):
                            f.write(f"Tarefa: {item['task']}\n  Conceitos Avaliados ({len(res_disp)} validados pelo Diretor de Arte):\n")
                            for idx_c, concept_c in enumerate(res_disp):
                                prompt_s = concept_c.get("image_prompt_selected", "N/A")
                                gen_r = concept_c.get("result")
                                is_b64_c = isinstance(gen_r,str) and len(gen_r)>100 and not gen_r.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', gen_r.strip())
                                status_c = "[IMAGEM BASE64 SELECIONADA]" if is_b64_c else str(gen_r)
                                f.write(f"    - Conceito {idx_c+1} (Prompt: {prompt_s[:100]}...): {status_c[:100]}...\n")
                        elif item['task'].startswith("TASK_GERAR_IMAGEM:"):
                             is_b64_like = isinstance(res_disp,str) and len(res_disp)>100 and not res_disp.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', res_disp.strip())
                             res_disp_str = f"[IMAGEM BASE64 - {len(res_disp)} chars]" if is_b64_like else str(res_disp)
                             f.write(f"Tarefa: {item['task']}\nResultado: {res_disp_str[:1000]}...\n")
                        else:
                            f.write(f"Tarefa: {item['task']}\nResultado: {str(res_disp)[:1000]}...\n")
                        f.write("\n")
                    f.write(f"--- VALIDAÇÃO FINAL ---\nTIPO: {output_type}\nCONTEÚDO (snippet para avaliação): {str(actual_main_content_to_save_for_assessment)[:1000]}...\nAVALIAÇÃO: {evaluation_text}\n")
                print_agent_message(agent_display_name, f"Avaliação salva: {assessment_file_name}")
            except Exception as e: print_agent_message(agent_display_name, f"Erro ao salvar avaliação: {e}\n{traceback.format_exc()}")

            # Salvando TODAS as imagens validadas e geradas com sucesso
            if validated_and_generated_images:
                log_message(f"Salvando {len(validated_and_generated_images)} imagem(ns) validada(s)...", agent_display_name)
                for idx, img_data in enumerate(validated_and_generated_images):
                    img_base64 = img_data["base64"]
                    img_prompt_slug = sanitize_filename(img_data["prompt"][:30])
                    prod_fname = os.path.join(OUTPUT_DIRECTORY, f"produto_imagem-png-base64_{goal_slug}_{CURRENT_TIMESTAMP_STR}_{idx+1}_{img_prompt_slug}.png")
                    try:
                        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', img_base64) and len(img_base64) % 4 == 0:
                            with open(prod_fname, "wb") as f_prod: f_prod.write(base64.b64decode(img_base64))
                            print_agent_message(agent_display_name, f"Produto (IMAGEM_PNG_BASE64 {idx+1}) salvo: {prod_fname}")
                        else:
                             print_agent_message(agent_display_name, f"Erro: Conteúdo para IMAGEM_PNG_BASE64 {idx+1} não parece ser base64 válido. Não salvo.")
                             log_message(f"Conteúdo para IMAGEM_PNG_BASE64 {idx+1} não é base64 válido.", agent_display_name)
                    except Exception as e: print_agent_message(agent_display_name, f"Erro ao salvar produto (IMAGEM_PNG_BASE64 {idx+1}): {e}\n{traceback.format_exc()}")
            
            # Salva um produto textual principal se não houver imagens OU se o tipo de saída principal for TEXTO_GERAL
            # e o conteúdo for válido.
            elif output_type == "TEXTO_GERAL" and \
                 actual_main_content_to_save_for_assessment and \
                 actual_main_content_to_save_for_assessment.strip() and \
                 actual_main_content_to_save_for_assessment != "[IMAGEM_BASE64_AQUI_PRINCIPAL]":
                
                prod_fname_text = os.path.join(OUTPUT_DIRECTORY, f"produto_texto-geral_{goal_slug}_{CURRENT_TIMESTAMP_STR}.txt")
                try:
                    with open(prod_fname_text, "w", encoding="utf-8") as f_prod:
                        f_prod.write(actual_main_content_to_save_for_assessment)
                    print_agent_message(agent_display_name, f"Produto (TEXTO_GERAL) salvo: {prod_fname_text}")
                except Exception as e:
                    print_agent_message(agent_display_name, f"Erro ao salvar produto textual: {e}")
            else:
                log_message(f"Nenhum produto principal salvo (sem imagens validadas ou conteúdo textual vazio/placeholder).", agent_display_name)

            print_agent_message(agent_display_name, "--- FIM DA VALIDAÇÃO ---")
        else: print_agent_message(agent_display_name, "Falha ao obter avaliação final da API.")


# --- Classe Worker ---
class Worker:
    # ... (código idêntico à v8.1)
    def __init__(self):
        self.gemini_model_name = GEMINI_TEXT_MODEL_NAME
        log_message("Instância do Worker criada.", "Worker")

    def execute_sub_task(self, sub_task_description, context_text_part, uploaded_file_objects):
        agent_display_name = "Worker"
        print_agent_message(agent_display_name, f"Executando: '{sub_task_description}'")
        
        prompt_text_for_worker = f"""
Você é um Agente Executor. Tarefa atual: "{sub_task_description}"
Contexto (resultados anteriores, objetivo original, arquivos):
{context_text_part}
Execute a tarefa. Se for "Criar uma descrição textual detalhada (prompt) para gerar a imagem de [...]", seu resultado DEVE ser APENAS essa descrição.
Se identificar NOVAS sub-tarefas cruciais, liste-as em 'NOVAS_TAREFAS_SUGERIDAS:' como array JSON de strings. Se não, omita.
Resultado da Tarefa:
"""
        prompt_parts = [prompt_text_for_worker] + uploaded_file_objects
        response_text = call_gemini_api_with_retry(
            prompt_parts,
            agent_display_name,
            model_name=self.gemini_model_name,
            gen_config=generation_config_text
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
                else:
                     lines = [ln.strip() for ln in potential_json_or_list.splitlines() if ln.strip() and not ln.startswith(('[', ']'))]
                     if lines: parsed_suggestions = lines
                
                if isinstance(parsed_suggestions, list):
                    for item in parsed_suggestions:
                        if isinstance(item, str):
                            sugg_tasks_strings.append(item.strip())
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str):
                             sugg_tasks_strings.append(item["tarefa"].strip())
                log_message(f"Novas tarefas sugeridas (strings filtradas): {sugg_tasks_strings}", agent_display_name)
            except Exception as e:
                log_message(f"Erro ao processar novas tarefas: {e}. Parte: {potential_json_or_list}\n{traceback.format_exc()}", agent_display_name)
        
        if task_res.lower().startswith("resultado da tarefa:"):
            task_res = task_res[len("resultado da tarefa:"):].strip()
        
        log_message(f"Resultado da sub-tarefa '{sub_task_description}': {task_res[:200]}...", agent_display_name)
        return task_res, sugg_tasks_strings

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v8.3"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Reutilização de Arquivos Cache) ---")
    print(f"📝 Logs: {LOG_FILE_NAME}\n📄 Saídas: {OUTPUT_DIRECTORY}\nℹ️ Cache Uploads: {UPLOADED_FILES_CACHE_DIR}")
    
    initial_goal_input = input("🎯 Defina a meta principal: ")
    print_user_message(initial_goal_input)
    
    # A função get_uploaded_files_info_from_user agora lida com cache e novos uploads
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()

    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
        log_message("Nenhuma meta definida. Encerrando.", "Sistema")
    else:
        manager = TaskManager()
        manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")



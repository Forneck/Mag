import google.generativeai as genai
import os
import json
import time
import datetime
import re
import traceback
import base64

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
UPLOADED_FILES_INFO_PATH = os.path.join(UPLOADED_FILES_CACHE_DIR, f"uploaded_files_info_{CURRENT_TIMESTAMP_STR}.json")

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

log_message(f"Modelo Gemini (texto/lógica): {GEMINI_TEXT_MODEL_NAME}", "Sistema")
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
def get_uploaded_files_info_from_user():
    uploaded_file_objects = []
    uploaded_files_metadata = []
    print_user_message("Adicionar arquivos complementares? (s/n)")
    if input("➡️ ").strip().lower() == 's':
        print_agent_message("Sistema", "Fazendo upload...")
        while True:
            print_user_message("Caminho do arquivo (ou 'fim'):")
            fp = input("➡️ ").strip()
            if fp.lower() == 'fim': break
            if not os.path.exists(fp) or not os.path.isfile(fp):
                print(f"❌ Arquivo '{fp}' inválido."); log_message(f"Arquivo '{fp}' inválido.", "Sistema"); continue
            try:
                dn = os.path.basename(fp)
                print_agent_message("Sistema", f"Upload de '{dn}'...")
                uf = genai.upload_file(path=fp, display_name=dn)
                uploaded_file_objects.append(uf)
                fm = {"user_path":fp,"display_name":uf.display_name,"file_id":uf.name,"uri":uf.uri,"mime_type":uf.mime_type,"size_bytes":uf.size_bytes,"state":str(uf.state)}
                uploaded_files_metadata.append(fm)
                print(f"✅ '{dn}' (ID: {uf.name}, Tipo: {uf.mime_type}) enviado!")
                log_message(f"Arquivo '{dn}' (ID: {uf.name}, URI: {uf.uri}, Tipo: {uf.mime_type}, Tamanho: {uf.size_bytes}B) enviado.", "Sistema")
            except Exception as e: print(f"❌ Erro no upload de '{fp}': {e}"); log_message(f"Erro no upload de '{fp}': {e}\n{traceback.format_exc()}", "Sistema")
    if uploaded_files_metadata:
        try:
            with open(UPLOADED_FILES_INFO_PATH, "w", encoding="utf-8") as f: json.dump(uploaded_files_metadata, f, indent=4)
            log_message(f"Metadados dos uploads salvos em: {UPLOADED_FILES_INFO_PATH}", "Sistema")
        except Exception as e: log_message(f"Erro ao salvar metadados dos uploads: {e}", "Sistema")
    return uploaded_file_objects, uploaded_files_metadata

def format_uploaded_files_info_for_prompt_text(files_metadata_list):
    if not files_metadata_list: return "Nenhum arquivo complementar fornecido."
    txt = "Arquivos complementares carregados (referencie pelo 'Nome de Exibição' ou 'ID do Arquivo'):\n"
    for m in files_metadata_list: txt += f"- Nome: {m['display_name']} (ID: {m['file_id']}, Tipo: {m['mime_type']})\n"
    return txt

# --- Classe ImageWorker ---
class ImageWorker:
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
3.  Após TODAS as tarefas de geração de imagem, adicionar UMA tarefa: "TASK_AVALIAR_IMAGENS: Avaliar as imagens/descrições geradas para [objetivo original] e selecionar a melhor."

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
                    # Garante que a lista de tarefas seja de strings
                    parsed_tasks = json.loads(json_str)
                    if isinstance(parsed_tasks, list) and all(isinstance(task, str) for task in parsed_tasks):
                        self.task_list = parsed_tasks
                    elif isinstance(parsed_tasks, list) and all(isinstance(task, dict) and "tarefa" in task for task in parsed_tasks):
                         self.task_list = [task_item["tarefa"] for task_item in parsed_tasks]
                         log_message("Decomposição retornou lista de dicionários, extraindo strings de 'tarefa'.", agent_display_name)
                    else:
                        raise ValueError("Formato de tarefa decomposta inesperado.")

                    log_message(f"Tarefas decompostas (strings): {self.task_list}", agent_display_name)
                    print_agent_message(agent_display_name, f"Tarefas decompostas: {self.task_list}")
                    return True
                else:
                    log_message(f"Decomposição não retornou JSON no formato esperado. Resposta: {response_text}", agent_display_name)
                    print_agent_message(agent_display_name, f"Decomposição não retornou JSON no formato esperado.")
            except json.JSONDecodeError as e:
                print_agent_message(agent_display_name, f"Erro ao decodificar JSON da decomposição: {e}. Resposta: {response_text}")
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
                else: # Fallback se não encontrar JSON delimitado
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


    def evaluate_and_select_image_concept(self, original_goal, image_task_results, uploaded_file_objects, files_metadata_for_prompt_text):
        agent_display_name = "Task Manager (Avaliação de Conceitos de Imagem)"
        # ... (código restante idêntico à v8.0)
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

Com base APENAS nos prompts usados e no sucesso/falha da geração, qual PROMPT DE IMAGEM você considera o mais promissor ou mais alinhado com o objetivo original?
Se múltiplas imagens foram geradas com sucesso, indique qual prompt levou à imagem que você recomendaria prosseguir.
Se todas falharam, ou se nenhuma tentativa foi feita, indique qual prompt parece o melhor para uma nova tentativa ou refinamento, ou se um novo prompt deve ser criado.
Se nenhuma tentativa foi feita e você não pode inferir um bom prompt, retorne "NENHUM_PROMPT_DEFINIDO".

Retorne APENAS o texto do prompt selecionado. Se nenhum for claramente superior ou se todos falharam de forma similar, retorne o prompt da primeira tentativa. Se nenhuma tentativa, retorne "NENHUM_PROMPT_DEFINIDO".

Prompt Selecionado:
"""
        prompt_parts_for_api = [prompt_text_part] + uploaded_file_objects
        selected_prompt_text_raw = call_gemini_api_with_retry(
            prompt_parts_for_api,
            agent_display_name,
            model_name=self.gemini_text_model_name,
            gen_config=generation_config_text
        )
        selected_prompt_text = (selected_prompt_text_raw or "").replace("Prompt Selecionado:", "").strip()

        if selected_prompt_text and selected_prompt_text != "NENHUM_PROMPT_DEFINIDO":
            print_agent_message(agent_display_name, f"Prompt de imagem selecionado para prosseguir: '{selected_prompt_text[:100]}...'")
            for res in image_task_results:
                if res.get('image_prompt_used', '').strip() == selected_prompt_text:
                    return selected_prompt_text, res.get("result")
            if image_task_results:
                 log_message(f"Prompt '{selected_prompt_text}' selecionado pelo LLM não corresponde a uma tentativa anterior ou é um refinamento. Não há resultado de imagem associado ainda.", agent_display_name)
                 return selected_prompt_text, "Nova tentativa necessária para este prompt selecionado."
            else:
                 log_message(f"Nenhuma tentativa de imagem anterior. LLM selecionou/sugeriu prompt: '{selected_prompt_text}'.", agent_display_name)
                 return selected_prompt_text, "Nova tentativa necessária para este prompt."

        if image_task_results:
            log_message("Fallback: LLM não selecionou um prompt claro, ou o prompt selecionado foi 'NENHUM_PROMPT_DEFINIDO'. Retornando o primeiro conceito de imagem das tentativas.", agent_display_name)
            return image_task_results[0].get('image_prompt_used'), image_task_results[0].get("result")
        
        log_message("Nenhuma tentativa de imagem e nenhum prompt selecionado pelo LLM.", agent_display_name)
        return None, "Falha na seleção do conceito de imagem (nenhuma tentativa e nenhum prompt definido)."


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
            # Garante que current_task_description seja uma string
            if isinstance(current_task_description, dict) and "tarefa" in current_task_description:
                current_task_description = current_task_description["tarefa"]
            elif not isinstance(current_task_description, str):
                log_message(f"Item de tarefa inválido encontrado: {current_task_description}. Pulando.", agent_display_name)
                current_task_index += 1
                continue


            print_agent_message(agent_display_name, f"Próxima tarefa ({current_task_index + 1}/{len(self.task_list)}): {current_task_description}")

            task_result_for_completed_list = None
            suggested_new_tasks_raw = [] # Para armazenar as sugestões brutas (podem ser dicts)
            
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
                selected_prompt, selected_generation_result = self.evaluate_and_select_image_concept(
                    initial_goal, image_generation_attempts, uploaded_file_objects, files_metadata_for_prompt_text
                )
                self.completed_tasks_results.append({
                    "task": current_task_description,
                    "result": selected_generation_result, 
                    "image_prompt_selected": selected_prompt
                })
                log_message(f"Tarefa '{current_task_description}' concluída. Conceito selecionado: '{selected_prompt}'. Resultado da geração: {str(selected_generation_result)[:100]}...", agent_display_name)
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

            if suggested_new_tasks_raw: # Agora suggested_new_tasks_raw é uma lista de strings do Worker
                print_agent_message(agent_display_name, f"Worker sugeriu: {suggested_new_tasks_raw}")
                # confirm_new_tasks_with_llm já retorna uma lista de strings
                approved_tasks_strings = self.confirm_new_tasks_with_llm(initial_goal, self.task_list, suggested_new_tasks_raw, uploaded_file_objects, files_metadata_for_prompt_text)
                if approved_tasks_strings:
                    for nt_idx, nt_string in enumerate(approved_tasks_strings):
                        if nt_string not in self.task_list: # Evita duplicatas se a tarefa já existir como string
                            # Insere novas tarefas após a atual, para serem processadas em seguida
                            self.task_list.insert(current_task_index + 1 + nt_idx, nt_string)
                            log_message(f"Nova tarefa APROVADA '{nt_string}' inserida na lista.", agent_display_name)
                    print_agent_message(agent_display_name, f"Lista de tarefas atualizada: {[str(t)[:100]+'...' for t in self.task_list]}") # Log resumido
                else: print_agent_message(agent_display_name, "Nenhuma nova tarefa sugerida aprovada.")
            
            current_task_index += 1; time.sleep(1) 

        print_agent_message(agent_display_name, "Todas as tarefas processadas.")
        self.validate_and_save_final_output(initial_goal, uploaded_file_objects, files_metadata_for_prompt_text)

    def extract_structured_output(self, llm_response_text):
        # ... (código restante idêntico à v8.0)
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
        # ... (código restante idêntico à v8.0)
        agent_display_name = "Task Manager (Validação)"
        print_agent_message(agent_display_name, "Validando resultado final...")
        if not self.completed_tasks_results:
            print_agent_message(agent_display_name, "Nenhuma tarefa concluída. Nada para validar ou salvar.")
            return

        results_summary_text = f"Meta Original: {original_goal}\nArquivos: {files_metadata_for_prompt_text}\nResultados Sub-tarefas:\n"
        final_image_base64_for_saving = None
        
        evaluation_task_result_obj = next((res for res in self.completed_tasks_results if res["task"].startswith("TASK_AVALIAR_IMAGENS:")), None)

        if evaluation_task_result_obj:
            selected_image_prompt = evaluation_task_result_obj.get("image_prompt_selected", "N/A")
            selected_image_generation_result = evaluation_task_result_obj.get("result")
            generation_status_text = "Falha na geração ou prompt não levou a uma imagem."
            
            if isinstance(selected_image_generation_result, str) and \
               len(selected_image_generation_result) > 100 and \
               not selected_image_generation_result.startswith("Falha") and \
               re.match(r'^[A-Za-z0-9+/]+={0,2}$', selected_image_generation_result.strip()):
                generation_status_text = "Sucesso (imagem base64 abaixo, se este for o produto final)."
                final_image_base64_for_saving = selected_image_generation_result.strip()
            elif selected_image_generation_result:
                generation_status_text = str(selected_image_generation_result)
            results_summary_text += f"- Tarefa: {evaluation_task_result_obj['task']}\n  Conceito Selecionado (Prompt): {selected_image_prompt}\n  Status da Geração do Conceito Selecionado: {generation_status_text}\n\n"
        else:
            for res in reversed(self.completed_tasks_results):
                if res["task"].startswith("TASK_GERAR_IMAGEM:"):
                    img_res = res.get("result")
                    if isinstance(img_res, str) and len(img_res) > 100 and not img_res.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', img_res.strip()):
                        final_image_base64_for_saving = img_res.strip()
                        results_summary_text += f"- Tarefa: {res['task']} (Última imagem gerada com sucesso selecionada como candidata)\n  Resultado: Sucesso (imagem base64 abaixo, se este for o produto final)\n\n"
                        break
        
        for item in self.completed_tasks_results:
            if not item["task"].startswith("TASK_AVALIAR_IMAGENS:"):
                result_display = item.get('result', 'N/A')
                if item['task'].startswith("TASK_GERAR_IMAGEM:"):
                    is_candidate_final_image = (isinstance(result_display, str) and result_display == final_image_base64_for_saving)
                    if isinstance(result_display, str) and len(result_display) > 100 and not result_display.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', result_display.strip()):
                        result_display = f"[IMAGEM GERADA - {'Candidata a produto final' if is_candidate_final_image else 'Outra variação/Não selecionada'}]"
                    else:
                        result_display = f"[TENTATIVA DE GERAR IMAGEM - Falhou: {result_display}]"
                results_summary_text += f"- Tarefa: {item['task']}\n  Resultado: {str(result_display)[:300]}...\n\n"

        prompt_text_part_validation = f"""
Você é um Gerenciador de Tarefas especialista em validação. Meta original: "{original_goal}"
Arquivos: {files_metadata_for_prompt_text}
Resultados das sub-tarefas:
{results_summary_text}
Com base nisso, sua tarefa é:
1.  Identificar o TIPO DE SAÍDA PRINCIPAL (TEXTO_GERAL, CODIGO_PYTHON, IMAGEM_PNG_BASE64, etc.).
2.  Fornecer o CONTEÚDO PRINCIPAL PARA SALVAR.
    - Se o tipo for IMAGEM_PNG_BASE64 e uma imagem foi gerada com sucesso e selecionada (indicado no resumo), use a placeholder "[IMAGEM_BASE64_AQUI]". O sistema substituirá.
    - Para outros tipos, forneça o conteúdo textual.
3.  Fornecer uma AVALIAÇÃO GERAL da execução e dos resultados.
Formato:
TIPO_DE_SAIDA_PRINCIPAL: [TIPO]
CONTEUDO_PRINCIPAL_PARA_SALVAR:
[Conteúdo textual ou "[IMAGEM_BASE64_AQUI]"]
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
            
            actual_main_content_to_save = main_content_from_llm
            if output_type == "IMAGEM_PNG_BASE64":
                if final_image_base64_for_saving:
                    actual_main_content_to_save = final_image_base64_for_saving
                    log_message("Usando base64 da imagem gerada/selecionada para CONTEUDO_PRINCIPAL.", agent_display_name)
                else:
                    output_type = "TEXTO_GERAL"
                    actual_main_content_to_save = f"Erro: LLM indicou saída de imagem, mas nenhuma imagem válida foi gerada/selecionada.\nConteúdo original do LLM: {main_content_from_llm if main_content_from_llm != '[IMAGEM_BASE64_AQUI]' else ''}"
                    evaluation_text += "\nNOTA: O tipo de saída foi alterado para TEXTO_GERAL pois nenhuma imagem válida estava disponível."
                    log_message("LLM indicou IMAGEM_PNG_BASE64, mas nenhuma imagem válida disponível. Revertendo para TEXTO_GERAL.", agent_display_name)

            print_agent_message(agent_display_name, f"Tipo Saída Final: {output_type}")
            print_agent_message(agent_display_name, f"Conteúdo Principal (Final para salvar): {str(actual_main_content_to_save)[:200]}...")
            print_agent_message(agent_display_name, f"Avaliação Geral: {evaluation_text}")

            goal_slug = sanitize_filename(original_goal)
            assessment_file_name = os.path.join(OUTPUT_DIRECTORY, f"avaliacao_completa_{goal_slug}_{CURRENT_TIMESTAMP_STR}.txt")
            try:
                with open(assessment_file_name, "w", encoding="utf-8") as f:
                    f.write(f"Meta: {original_goal}\nArquivos: {files_metadata_for_prompt_text}\n--- RESULTADOS SUB-TAREFAS ---\n")
                    for item in self.completed_tasks_results:
                        res_disp = item.get('result', 'N/A')
                        is_base64_like = isinstance(res_disp,str) and len(res_disp)>100 and not res_disp.startswith("Falha") and re.match(r'^[A-Za-z0-9+/]+={0,2}$', res_disp.strip())
                        
                        if item['task'].startswith("TASK_GERAR_IMAGEM:") and is_base64_like:
                             res_disp = f"[IMAGEM BASE64 - {len(res_disp)} chars]"
                        elif item['task'].startswith("TASK_AVALIAR_IMAGENS:") and is_base64_like:
                             res_disp = f"[IMAGEM BASE64 SELECIONADA - {len(res_disp)} chars]"

                        f.write(f"Tarefa: {item['task']}\nResultado: {str(res_disp)[:1000]}...\n")
                        if item['task'].startswith("TASK_AVALIAR_IMAGENS:") and "image_prompt_selected" in item:
                            f.write(f"  Prompt Selecionado: {item['image_prompt_selected']}\n")
                        f.write("\n")
                    f.write(f"--- VALIDAÇÃO FINAL ---\nTIPO: {output_type}\nCONTEÚDO (snippet): {str(actual_main_content_to_save)[:1000]}...\nAVALIAÇÃO: {evaluation_text}\n")
                print_agent_message(agent_display_name, f"Avaliação salva: {assessment_file_name}")
            except Exception as e: print_agent_message(agent_display_name, f"Erro ao salvar avaliação: {e}\n{traceback.format_exc()}")

            file_ext_map = {"CODIGO_PYTHON":".py","CODIGO_HTML":".html","IMAGEM_PNG_BASE64":".png", "TEXTO_GERAL": ".txt"}
            default_ext = ".txt"
            ext = file_ext_map.get(output_type, default_ext)
            prod_fname_base = f"produto_{output_type.lower().replace('_','-')}_{goal_slug}_{CURRENT_TIMESTAMP_STR}"
            
            if output_type == "TEXTO_GERAL" and (not actual_main_content_to_save or actual_main_content_to_save == "[IMAGEM_BASE64_AQUI]"):
                 log_message(f"Produto ({output_type}) não salvo pois o conteúdo está vazio ou é placeholder.", agent_display_name)
            elif actual_main_content_to_save and isinstance(actual_main_content_to_save, str) and actual_main_content_to_save.strip():
                prod_fname = os.path.join(OUTPUT_DIRECTORY, f"{prod_fname_base}{ext}")
                try:
                    if output_type == "IMAGEM_PNG_BASE64":
                        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', actual_main_content_to_save) and len(actual_main_content_to_save) % 4 == 0:
                            with open(prod_fname, "wb") as f_prod: f_prod.write(base64.b64decode(actual_main_content_to_save))
                            print_agent_message(agent_display_name, f"Produto ({output_type}) salvo: {prod_fname}")
                        else:
                             print_agent_message(agent_display_name, f"Erro: Conteúdo para IMAGEM_PNG_BASE64 não parece ser base64 válido. Não salvo. Conteúdo: {actual_main_content_to_save[:100]}...")
                             log_message(f"Conteúdo para IMAGEM_PNG_BASE64 não é base64 válido.", agent_display_name)
                    else:
                        with open(prod_fname, "w", encoding="utf-8") as f_prod: f_prod.write(actual_main_content_to_save)
                        print_agent_message(agent_display_name, f"Produto ({output_type}) salvo: {prod_fname}")
                except Exception as e: print_agent_message(agent_display_name, f"Erro ao salvar produto ({output_type}): {e}\n{traceback.format_exc()}")
            else:
                 log_message(f"Produto ({output_type}) não salvo pois o conteúdo está vazio ou não é string.", agent_display_name)
            print_agent_message(agent_display_name, "--- FIM DA VALIDAÇÃO ---")
        else: print_agent_message(agent_display_name, "Falha ao obter avaliação final da API.")


# --- Classe Worker ---
class Worker:
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

        task_res, sugg_tasks_strings = response_text, [] # Garante que sugg_tasks_strings seja uma lista de strings
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
                else: # Fallback para linhas se não for JSON claro
                     lines = [ln.strip() for ln in potential_json_or_list.splitlines() if ln.strip() and not ln.startswith('[') and not ln.startswith(']')]
                     if lines: parsed_suggestions = lines
                
                # Garante que sugg_tasks_strings seja uma lista de strings
                if isinstance(parsed_suggestions, list):
                    for item in parsed_suggestions:
                        if isinstance(item, str):
                            sugg_tasks_strings.append(item.strip())
                        elif isinstance(item, dict) and "tarefa" in item and isinstance(item["tarefa"], str):
                             sugg_tasks_strings.append(item["tarefa"].strip())
                        # Ignora outros formatos
                log_message(f"Novas tarefas sugeridas (strings filtradas): {sugg_tasks_strings}", agent_display_name)
            except Exception as e:
                log_message(f"Erro ao processar novas tarefas: {e}. Parte: {potential_json_or_list}\n{traceback.format_exc()}", agent_display_name)
        
        if task_res.lower().startswith("resultado da tarefa:"):
            task_res = task_res[len("resultado da tarefa:"):].strip()
        
        log_message(f"Resultado da sub-tarefa '{sub_task_description}': {task_res[:200]}...", agent_display_name)
        return task_res, sugg_tasks_strings

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v8.1"
    log_message(f"--- Início da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION} - Correção AttributeError dict.startswith) ---")
    print(f"📝 Logs: {LOG_FILE_NAME}\n📄 Saídas: {OUTPUT_DIRECTORY}\nℹ️ Cache Uploads: {UPLOADED_FILES_CACHE_DIR}")
    
    initial_goal_input = input("🎯 Defina a meta principal: ")
    print_user_message(initial_goal_input)
    uploaded_files, uploaded_files_meta = get_uploaded_files_info_from_user()

    if not initial_goal_input.strip():
        print("Nenhuma meta definida. Encerrando.")
        log_message("Nenhuma meta definida. Encerrando.", "Sistema")
    else:
        manager = TaskManager()
        manager.run_workflow(initial_goal_input, uploaded_files, uploaded_files_meta)

    log_message(f"--- Fim da Execução ({SCRIPT_VERSION}) ---", "Sistema")
    print("\n--- Fim da Execução ---")

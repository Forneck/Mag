
import google.generativeai as genai
from google.generativeai import types
import os
import json
import time
import datetime
import re
import traceback
import glob
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import urllib.parse

# --- Configuração dos Diretórios e Arquivos ---
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_agent_logs")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "gemini_final_outputs")

for directory in [LOG_DIRECTORY, OUTPUT_DIRECTORY]:
    if not os.path.exists(directory):
        os.makedirs(directory)

CURRENT_TIMESTAMP_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIRECTORY, f"agent_log_{CURRENT_TIMESTAMP_STR}.txt")

# --- Constantes ---
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_FACTOR = 2

# --- Modelos Gemini ---
# Updated to latest Gemini 2.5 preview models
GEMINI_TEXT_MODEL_NAME = "gemini-2.5-flash-preview"
GEMINI_IMAGE_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

# --- Configurações de Segurança Gemini ---
safety_settings_gemini=[
    {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},
    {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_MEDIUM_AND_ABOVE"},
    {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_NONE"},
    {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_NONE"}
]

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
    full_log_message = f"[{timestamp}] [{source}]: {message}\\n"
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

# --- Ferramentas para o Agente ---
def save_file(filename: str, content: str) -> dict:
    """Salva o conteúdo textual fornecido em um arquivo com o nome especificado."""
    try:
        sanitized_fn = sanitize_filename(filename)
        if not sanitized_fn: return {"status": "error", "message": "O nome do arquivo é inválido."}
        full_path = os.path.join(OUTPUT_DIRECTORY, sanitized_fn)
        with open(full_path, "w", encoding="utf-8") as f: f.write(content)
        log_message(f"Arquivo '{sanitized_fn}' salvo.", "Tool:save_file")
        return {"status": "success", "message": f"Arquivo '{sanitized_fn}' salvo."}
    except Exception as e:
        log_message(f"Erro ao salvar arquivo '{filename}': {e}", "Tool:save_file")
        return {"status": "error", "message": f"Erro ao salvar arquivo: {str(e)}"}

def generate_image(image_prompt_in_english: str, base_image_path: Optional[str] = None) -> dict:
    """Gera ou edita uma imagem a partir de um prompt em inglês."""
    try:
        log_message(f"Gerando imagem: '{image_prompt_in_english[:100]}...'", "Tool:generate_image")
        contents = [image_prompt_in_english]
        if base_image_path and os.path.exists(base_image_path):
            log_message(f"Usando imagem base: {base_image_path}", "Tool:generate_image")
            contents.append(Image.open(base_image_path))
        
        model = genai.GenerativeModel(GEMINI_IMAGE_MODEL_NAME)
        response = model.generate_content(
            contents=contents,
            safety_settings=safety_settings_gemini # Adicionada as safety_settings aqui
        )
        image_part = next((p for p in response.candidates[0].content.parts if hasattr(p, 'inline_data') and p.inline_data), None)
        image_bytes = image_part.inline_data.data if image_part else None
        if not image_bytes: return {"status": "error", "message": "API não retornou imagem."}
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"imagem_{sanitize_filename(image_prompt_in_english[:20])}_{ts}.png"
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)
        with open(full_path, "wb") as f: f.write(image_bytes)
        log_message(f"Imagem salva: '{filename}'.", "Tool:generate_image")
        return {"status": "success", "message": f"Imagem salva como '{filename}'."}
    except Exception as e:
        log_message(f"Erro em generate_image: {e}\\n{traceback.format_exc()}", "Tool:generate_image")
        return {"status": "error", "message": f"Erro ao gerar imagem: {e}"}

def generate_video(video_prompt_in_english: str, duration_seconds: int = 5) -> dict:
    """Gera um vídeo a partir de um prompt em inglês (Veo3 - quando disponível)."""
    try:
        log_message(f"Tentando gerar vídeo: '{video_prompt_in_english[:100]}...'", "Tool:generate_video")
        
        # NOTE: This is a placeholder for when Veo3 becomes available in the API
        # For now, we'll document the request and return a planning response
        
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"video_plan_{sanitize_filename(video_prompt_in_english[:20])}_{ts}.txt"
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        plan_content = f"""PLANO DE VÍDEO PARA IMPLEMENTAÇÃO FUTURA (Veo3)
=================================================

Prompt: {video_prompt_in_english}
Duração solicitada: {duration_seconds} segundos
Timestamp: {datetime.datetime.now().isoformat()}

STATUS: Aguardando disponibilidade da API Veo3 do Google

ESPECIFICAÇÕES TÉCNICAS PLANEJADAS:
- Modelo: Veo3 (quando disponível)
- Resolução: 1080p
- Duração: {duration_seconds}s
- Formato: MP4

PRÓXIMOS PASSOS:
1. Monitorar atualizações da API do Google
2. Implementar integração com Veo3
3. Adicionar configurações de qualidade de vídeo
4. Implementar pós-processamento se necessário

Este arquivo documenta a solicitação para implementação futura.
"""
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(plan_content)
        
        log_message(f"Plano de vídeo documentado: '{filename}'.", "Tool:generate_video")
        return {
            "status": "planned", 
            "message": f"Vídeo planejado e documentado em '{filename}'. Implementação aguarda API Veo3."
        }
        
    except Exception as e:
        log_message(f"Erro em generate_video: {e}\\n{traceback.format_exc()}", "Tool:generate_video")
        return {"status": "error", "message": f"Erro ao planejar vídeo: {e}"}

def google_search(query: str, num_results: int = 5) -> dict:
    """Realiza uma busca no Google e retorna os resultados com títulos e links."""
    try:
        log_message(f"Buscando no Google: '{query}' (máximo {num_results} resultados)", "Tool:google_search")
        
        search_results = []
        for result in search(query, num_results=num_results, sleep_interval=2):
            search_results.append(result)
            if len(search_results) >= num_results:
                break
            
        if not search_results:
            return {"status": "success", "message": "Nenhum resultado encontrado.", "results": []}
        
        # Try to get titles by fetching the pages
        detailed_results = []
        for i, url in enumerate(search_results):
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title')
                    title_text = title.text.strip() if title else f"Resultado {i+1}"
                    detailed_results.append({
                        "title": title_text,
                        "url": url,
                        "snippet": f"Link {i+1} - {url}"
                    })
                else:
                    detailed_results.append({
                        "title": f"Resultado {i+1}",
                        "url": url,
                        "snippet": f"Status: {response.status_code}"
                    })
            except Exception as e:
                detailed_results.append({
                    "title": f"Resultado {i+1}",
                    "url": url,
                    "snippet": f"Erro ao acessar: {str(e)}"
                })
        
        result_text = f"Encontrados {len(detailed_results)} resultados para '{query}':\\n"
        for result in detailed_results:
            result_text += f"\\n• {result['title']}\\n  URL: {result['url']}\\n  {result['snippet']}\\n"
        
        log_message(f"Busca concluída: {len(detailed_results)} resultados", "Tool:google_search")
        
        return {
            "status": "success", 
            "message": result_text,
            "results": detailed_results,
            "count": len(detailed_results)
        }
        
    except Exception as e:
        log_message(f"Erro em google_search: {e}\\n{traceback.format_exc()}", "Tool:google_search")
        return {"status": "error", "message": f"Erro ao buscar no Google: {e}"}

def fetch_webpage_content(url: str, extract_text_only: bool = True) -> dict:
    """Busca o conteúdo de uma página web e extrai texto ou HTML."""
    try:
        log_message(f"Buscando conteúdo da página: {url}", "Tool:fetch_webpage_content")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if extract_text_only:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = '\\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit content size
            if len(text_content) > 8000:
                text_content = text_content[:8000] + "\\n\\n[CONTEÚDO TRUNCADO...]"
            
            # Get page title
            title = soup.find('title')
            page_title = title.text.strip() if title else "Título não encontrado"
            
            result_message = f"Conteúdo extraído de: {url}\\n\\nTítulo: {page_title}\\n\\nConteúdo:\\n{text_content}"
            
            log_message(f"Texto extraído com sucesso de {url} - {len(text_content)} caracteres", "Tool:fetch_webpage_content")
            
            return {
                "status": "success",
                "message": result_message,
                "title": page_title,
                "content": text_content,
                "url": url
            }
        else:
            # Return raw HTML (limited)
            html_content = response.text
            if len(html_content) > 10000:
                html_content = html_content[:10000] + "\\n\\n[HTML TRUNCADO...]"
            
            return {
                "status": "success",
                "message": f"HTML bruto extraído de: {url}\\n\\n{html_content}",
                "content": html_content,
                "url": url
            }
            
    except requests.exceptions.RequestException as e:
        log_message(f"Erro de requisição em fetch_webpage_content: {e}", "Tool:fetch_webpage_content")
        return {"status": "error", "message": f"Erro ao acessar a página: {e}"}
    except Exception as e:
        log_message(f"Erro em fetch_webpage_content: {e}\\n{traceback.format_exc()}", "Tool:fetch_webpage_content")
        return {"status": "error", "message": f"Erro ao processar página: {e}"}

def browser_automation(action: str, url: str = "", element_selector: str = "", text_input: str = "", wait_seconds: int = 3) -> dict:
    """Automação básica de browser usando requests e BeautifulSoup (versão simplificada sem Playwright)."""
    try:
        log_message(f"Automação browser: ação '{action}' em {url}", "Tool:browser_automation")
        
        if action == "navigate":
            if not url:
                return {"status": "error", "message": "URL é obrigatória para navegação"}
            
            result = fetch_webpage_content(url, extract_text_only=True)
            if result["status"] == "success":
                return {
                    "status": "success",
                    "message": f"Navegação para {url} concluída.\\n\\n{result['message']}",
                    "content": result.get("content", ""),
                    "title": result.get("title", "")
                }
            else:
                return result
                
        elif action == "search_content":
            if not url or not element_selector:
                return {"status": "error", "message": "URL e texto de busca são obrigatórios"}
            
            # Use element_selector as search text
            search_text = element_selector.lower()
            result = fetch_webpage_content(url, extract_text_only=True)
            
            if result["status"] == "success":
                content = result.get("content", "").lower()
                if search_text in content:
                    # Find context around the search term
                    index = content.find(search_text)
                    start = max(0, index - 200)
                    end = min(len(content), index + 200)
                    context = result.get("content", "")[start:end]
                    
                    return {
                        "status": "success",
                        "message": f"Texto '{element_selector}' encontrado em {url}:\\n\\n...{context}...",
                        "found": True,
                        "context": context
                    }
                else:
                    return {
                        "status": "success",
                        "message": f"Texto '{element_selector}' não encontrado em {url}",
                        "found": False
                    }
            else:
                return result
                
        elif action == "extract_links":
            if not url:
                return {"status": "error", "message": "URL é obrigatória para extrair links"}
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urllib.parse.urljoin(url, href)
                elif not href.startswith('http'):
                    continue
                    
                if text and len(text) > 0:
                    links.append({"text": text, "url": href})
            
            # Limit number of links
            links = links[:20]
            
            links_text = f"Links extraídos de {url}:\\n\\n"
            for i, link in enumerate(links, 1):
                links_text += f"{i}. {link['text']}\\n   URL: {link['url']}\\n\\n"
            
            return {
                "status": "success",
                "message": links_text,
                "links": links,
                "count": len(links)
            }
            
        else:
            return {
                "status": "error", 
                "message": f"Ação '{action}' não suportada. Use: navigate, search_content, extract_links"
            }
            
    except requests.exceptions.RequestException as e:
        log_message(f"Erro de requisição em browser_automation: {e}", "Tool:browser_automation")
        return {"status": "error", "message": f"Erro de rede: {e}"}
    except Exception as e:
        log_message(f"Erro em browser_automation: {e}\\n{traceback.format_exc()}", "Tool:browser_automation")
        return {"status": "error", "message": f"Erro na automação: {e}"}


# Define tools using the new API format
save_file_tool = genai.types.FunctionDeclaration(
    name="save_file",
    description="Salva o conteúdo textual fornecido em um arquivo com o nome especificado",
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Nome do arquivo a ser salvo"
            },
            "content": {
                "type": "string", 
                "description": "Conteúdo textual a ser salvo no arquivo"
            }
        },
        "required": ["filename", "content"]
    }
)

generate_image_tool = genai.types.FunctionDeclaration(
    name="generate_image",
    description="Gera ou edita uma imagem a partir de um prompt em inglês",
    parameters={
        "type": "object",
        "properties": {
            "image_prompt_in_english": {
                "type": "string",
                "description": "Prompt em inglês para gerar a imagem"
            },
            "base_image_path": {
                "type": "string",
                "description": "Caminho opcional para uma imagem base para edição"
            }
        },
        "required": ["image_prompt_in_english"]
    }
)

generate_video_tool = genai.types.FunctionDeclaration(
    name="generate_video",
    description="Gera um vídeo a partir de um prompt em inglês (Veo3 - planejamento para implementação futura)",
    parameters={
        "type": "object",
        "properties": {
            "video_prompt_in_english": {
                "type": "string",
                "description": "Prompt em inglês para gerar o vídeo"
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Duração do vídeo em segundos (padrão: 5)"
            }
        },
        "required": ["video_prompt_in_english"]
    }
)

google_search_tool = genai.types.FunctionDeclaration(
    name="google_search",
    description="Realiza uma busca no Google e retorna os resultados com títulos e links",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Termo de busca para o Google"
            },
            "num_results": {
                "type": "integer",
                "description": "Número máximo de resultados a retornar (padrão: 5)"
            }
        },
        "required": ["query"]
    }
)

fetch_webpage_content_tool = genai.types.FunctionDeclaration(
    name="fetch_webpage_content",
    description="Busca o conteúdo de uma página web e extrai texto limpo ou HTML bruto",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL da página web a ser acessada"
            },
            "extract_text_only": {
                "type": "boolean",
                "description": "Se true, extrai apenas texto limpo. Se false, retorna HTML bruto (padrão: true)"
            }
        },
        "required": ["url"]
    }
)

browser_automation_tool = genai.types.FunctionDeclaration(
    name="browser_automation",
    description="Automação básica de browser para navegar, buscar conteúdo e extrair links de páginas web",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Ação a realizar: 'navigate' (navegar para URL), 'search_content' (buscar texto na página), 'extract_links' (extrair todos os links)",
                "enum": ["navigate", "search_content", "extract_links"]
            },
            "url": {
                "type": "string",
                "description": "URL da página web (obrigatório para todas as ações)"
            },
            "element_selector": {
                "type": "string",
                "description": "Texto a buscar na página (usado com action='search_content')"
            },
            "text_input": {
                "type": "string",
                "description": "Texto para entrada (reservado para futuras funcionalidades)"
            },
            "wait_seconds": {
                "type": "integer",
                "description": "Segundos para aguardar (padrão: 3)"
            }
        },
        "required": ["action", "url"]
    }
)

AVAILABLE_TOOLS = {
    "save_file": save_file, 
    "generate_image": generate_image, 
    "generate_video": generate_video,
    "google_search": google_search,
    "fetch_webpage_content": fetch_webpage_content,
    "browser_automation": browser_automation
}

AVAILABLE_TOOL_DECLARATIONS = [
    save_file_tool, 
    generate_image_tool, 
    generate_video_tool,
    google_search_tool,
    fetch_webpage_content_tool,
    browser_automation_tool
]

# --- Funções de Comunicação e Arquivos ---
def print_agent_message(agent_name, message): print(f"\n🤖 [{agent_name}]: {message}"); log_message(message, agent_name)
def print_user_message(message): print(f"\n👤 [Usuário]: {message}"); log_message(message, "Usuário")
def print_thought_message(message): print(f"\n🧠 [Pensamento]:\\n{message}"); log_message(f"PENSAMENTO:\n{message}", "Agente")

def get_uploaded_files_info_from_user():
    uploaded_file_objects, uploaded_files_metadata = [], []
    try:
        print_agent_message("Sistema", "Verificando arquivos na API...")
        api_files_list = list(genai.list_files())
        log_message(f"Encontrados {len(api_files_list)} arquivos na API.")
        if api_files_list:
            print_agent_message("Sistema", f"Encontrados {len(api_files_list)} arquivos existentes.")
            if input("👤 Deseja limpar TODOS os arquivos da API? (s/n) ➡️ ").lower() == 's':
                print_agent_message("Sistema", "Limpando arquivos...")
                for file_obj in api_files_list:
                    try: genai.delete_file(name=file_obj.name); time.sleep(0.2)
                    except Exception as e: log_message(f"Falha ao deletar {file_obj.name}: {e}", "Sistema")
                print_agent_message("Sistema", "Limpeza concluída."); api_files_list = []                                                        
            if api_files_list:
                print_agent_message("Sistema", "Arquivos restantes na API:")
                for i, f in enumerate(api_files_list):
                    display_name_to_show = f.display_name if f.display_name else f.name
                    print(f"  {i+1}. {display_name_to_show}")
                if input("👤 Reutilizar arquivos existentes? (s/n) ➡️ ").lower() == 's':
                    choices = input("👤 Números (ex: 1,3) ou 'todos': ➡️ ").lower()
                    sel_indices = list(range(len(api_files_list))) if choices == 'todos' else [int(x.strip()) - 1 for x in choices.split(',') if x.strip().isdigit()]
                    for idx in sel_indices:
                        if 0 <= idx < len(api_files_list):
                            file_obj = api_files_list[idx]
                            uploaded_file_objects.append(file_obj)
                            meta_display_name = file_obj.display_name if file_obj.display_name else file_obj.name
                            uploaded_files_metadata.append({"file_id": file_obj.name, "display_name": meta_display_name})
                            print_agent_message("Sistema", f"✅ '{meta_display_name}' selecionado.")
    except Exception as e:
        log_message(f"Erro ao gerenciar arquivos da API: {e}", "Sistema")
        print_agent_message("Sistema", "AVISO: Não foi possível gerenciar arquivos da API.")

    if input("👤 Fazer upload de novos arquivos? (s/n) (Suporta curingas como *.txt, pasta/*.md) ➡️ ").lower() == 's':
        while True:
            file_pattern = input("👤 Caminho do arquivo ou padrão (ex: *.txt, pasta/*.md) (ou 'fim'): ➡️ ").strip()
            if file_pattern.lower() == 'fim': break

            # Expandir o padrão usando glob
            found_files = glob.glob(file_pattern)

            if not found_files:
                print_agent_message("Sistema", f"❌ Nenhum arquivo encontrado para o padrão: '{file_pattern}'. Tente novamente.")
                continue

            print_agent_message("Sistema", f"Encontrados {len(found_files)} arquivo(s) para o padrão '{file_pattern}':")
            for f_path in found_files:
                print(f"  - {f_path}")

            if input("👤 Confirmar upload dos arquivos encontrados? (s/n) ➡️ ").lower() != 's':
                print_agent_message("Sistema", "Upload cancelado para este padrão.")
                continue

            for fp in found_files:
                dn = os.path.basename(fp)
                try:
                    if os.path.isfile(fp): # Adicional checagem caso o glob retorne algo que não é um arquivo direto
                        print_agent_message("Sistema", f"Enviando '{dn}'...")
                        file_obj = genai.upload_file(path=fp)
                        uploaded_file_objects.append(file_obj)
                        uploaded_files_metadata.append({"file_id": file_obj.name, "display_name": dn})
                        print_agent_message("Sistema", f"✅ '{dn}' enviado."); time.sleep(0.5) # Pequena pausa para evitar sobrecarga da API
                    else:
                        print_agent_message("Sistema", f"ℹ️ '{fp}' não é um arquivo válido e será ignorado.")
                except Exception as e:
                    print_agent_message("Sistema", f"❌ Erro no upload de '{dn}': {e}")
                    log_message(f"Erro no upload de '{dn}': {e}", "Sistema")
            print_agent_message("Sistema", f"Concluído o processamento do padrão '{file_pattern}'.")
    return uploaded_file_objects, uploaded_files_metadata

def call_gemini_api_with_retry(prompt_parts, agent_name="Sistema", gen_config_dict=None):
    log_message(f"Chamando API para {agent_name}...", "Sistema")
    
    if gen_config_dict is None:
        gen_config_dict = {}
    
    # Adiciona as safety_settings ao dicionário gen_config_dict se não estiver presente
    if 'safety_settings' not in gen_config_dict:
        gen_config_dict['safety_settings'] = safety_settings_gemini

    # Create the model instance
    model = genai.GenerativeModel(
        model_name=GEMINI_TEXT_MODEL_NAME,
        safety_settings=gen_config_dict.pop('safety_settings', safety_settings_gemini)
    )
    
    # Remove safety_settings from gen_config_dict since it's passed to model
    generation_config = genai.GenerationConfig(**{k: v for k, v in gen_config_dict.items() if k != 'tools'})
    tools = gen_config_dict.get('tools', None)
    
    log_message(f"Usando config: {generation_config}", "Sistema")
    
    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_API_RETRIES):
        log_message(f"Tentativa {attempt + 1}/{MAX_API_RETRIES}...", "Sistema")
        try:
            if tools:
                response = model.generate_content(
                    contents=prompt_parts,
                    generation_config=generation_config,
                    tools=tools
                )
            else:
                response = model.generate_content(
                    contents=prompt_parts,
                    generation_config=generation_config
                )
            return response
        except Exception as e:
            log_message(f"Exceção: {type(e).__name__} - {e}\\n{traceback.format_exc()}", "Sistema")
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(current_retry_delay)
                current_retry_delay *= RETRY_BACKOFF_FACTOR
            else: return None
    return None

def extract_and_print_thoughts(response):
    """Extrai e exibe pensamentos/raciocínio do modelo Gemini."""
    if not response or not response.candidates:
        return
        
    for candidate in response.candidates:
        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
            for part in candidate.content.parts:
                # Look for thinking/thought content in different possible formats
                if hasattr(part, 'thought') and part.thought:
                    if hasattr(part.thought, 'content'):
                        print_thought_message(part.thought.content)
                    elif hasattr(part, 'text') and part.text:
                        print_thought_message(part.text)
                elif hasattr(part, 'text') and part.text and 'pensamento:' in part.text.lower():
                    # Extract thinking content from text
                    lines = part.text.split('\n')
                    thought_lines = []
                    in_thought = False
                    for line in lines:
                        if 'pensamento:' in line.lower() or 'thinking:' in line.lower():
                            in_thought = True
                            thought_lines.append(line)
                        elif in_thought and (line.strip() == '' or line.startswith('---')):
                            break
                        elif in_thought:
                            thought_lines.append(line)
                    
                    if thought_lines:
                        print_thought_message('\n'.join(thought_lines))

# --- Classes dos Agentes ---

class RouterAgent:
    """Router agent que decide qual agente especializado usar para cada tarefa."""
    
    def __init__(self):
        self.routing_instruction = (
            "Você é um Router Agent especialista. Analise a tarefa fornecida e determine qual tipo de agente "
            "é mais adequado para executá-la. Responda com um JSON contendo 'agent_type' e 'reasoning'. "
            "Tipos disponíveis: 'text_worker' (tarefas de texto/código simples), 'image_worker' (geração de imagens), "
            "'video_worker' (geração de vídeos), 'analysis_worker' (análise de dados), "
            "'thinking_worker' (raciocínio complexo, problemas difíceis, planejamento estratégico), "
            "'browser_worker' (busca no Google, navegação web, extração de conteúdo de sites, automação de browser). "
            "Exemplo: {'agent_type': 'browser_worker', 'reasoning': 'Tarefa requer busca web ou navegação'}"
        )
        log_message("RouterAgent criado.", "RouterAgent")
    
    def route_task(self, task_description, context=""):
        """Decide qual agente deve executar a tarefa."""
        print_agent_message("RouterAgent", f"Analisando roteamento para: '{task_description}'")
        
        prompt_parts = [
            f"{self.routing_instruction}\n\n"
            f"Contexto: {context}\n"
            f"Tarefa a ser roteada: '{task_description}'\n\n"
            f"Determine o melhor agente para esta tarefa."
        ]
        
        gen_config = {
            "temperature": 0.3,
            "response_mime_type": "application/json"
        }
        
        response = call_gemini_api_with_retry(prompt_parts, "RouterAgent", gen_config_dict=gen_config)
        
        if not response or not response.text:
            log_message("Router falhou, usando text_worker como padrão", "RouterAgent")
            return "text_worker", "Fallback para texto devido a falha no roteamento"
        
        try:
            text_response = response.text.strip()
            if text_response.startswith("```json"): 
                text_response = text_response[7:-3].strip()
            route_dict = json.loads(text_response)
            
            agent_type = route_dict.get("agent_type", "text_worker")
            reasoning = route_dict.get("reasoning", "Sem justificativa fornecida")
            
            print_agent_message("RouterAgent", f"Roteado para: {agent_type} - {reasoning}")
            return agent_type, reasoning
            
        except (json.JSONDecodeError, TypeError) as e:
            log_message(f"Erro no parsing do router: {e}. Resposta: '{response.text}'", "RouterAgent")
            return "text_worker", "Fallback para texto devido a erro de parsing"

class Worker:
    def __init__(self, task_manager):
        self.task_manager = task_manager
        log_message("Worker (v11.26 - Gemini 2.5) criado.", "Worker")

    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "Worker"
        print_agent_message(agent_name, f"Executando: '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
                                    f"Objetivo Geral: {original_goal}\n\n"
                                    f"Sua tarefa agora: \\'{task_description}\\'. ")
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", "Worker")
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", "Worker")
        
        return {"text_content": response.text.strip() if response.text else "Ação concluída."}, []

class ImageWorker(Worker):
    """Agente especializado em tarefas relacionadas a imagens."""
    
    def __init__(self, task_manager):
        super().__init__(task_manager)
        log_message("ImageWorker (v11.26) criado.", "ImageWorker")
    
    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "ImageWorker"
        print_agent_message(agent_name, f"Executando (imagem): '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(
            f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
            f"Objetivo Geral: {original_goal}\n\n"
            f"TAREFA DE IMAGEM: {task_description}\n"
            f"Foque especificamente em gerar, editar ou analisar imagens. "
            f"Use a função generate_image quando apropriado."
        )
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS,
            "temperature": 0.7  # Mais criatividade para imagens
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", agent_name)
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", agent_name)
        
        return {"text_content": response.text.strip() if response.text else "Imagem processada."}, []

class AnalysisWorker(Worker):
    """Agente especializado em análise e pensamento complexo."""
    
    def __init__(self, task_manager):
        super().__init__(task_manager)
        log_message("AnalysisWorker (v11.26) criado.", "AnalysisWorker")
    
    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "AnalysisWorker"
        print_agent_message(agent_name, f"Executando (análise): '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(
            f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
            f"Objetivo Geral: {original_goal}\n\n"
            f"TAREFA DE ANÁLISE: {task_description}\n"
            f"Pense profundamente sobre esta tarefa. Considere múltiplas perspectivas, "
            f"analise dados e forneça insights detalhados. Use raciocínio estruturado."
        )
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS,
            "temperature": 0.3  # Mais precisão para análise
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", agent_name)
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", agent_name)
        
        return {"text_content": response.text.strip() if response.text else "Análise concluída."}, []

class VideoWorker(Worker):
    """Agente especializado em tarefas relacionadas a vídeos (Veo3 quando disponível)."""
    
    def __init__(self, task_manager):
        super().__init__(task_manager)
        log_message("VideoWorker (v11.26 - Veo3 ready) criado.", "VideoWorker")
    
    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "VideoWorker"
        print_agent_message(agent_name, f"Executando (vídeo): '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(
            f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
            f"Objetivo Geral: {original_goal}\n\n"
            f"TAREFA DE VÍDEO: {task_description}\n"
            f"NOTA: Funcionalidade de vídeo (Veo3) ainda não implementada na API. "
            f"Por ora, documente os requisitos e planeje a implementação futura."
        )
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS,
            "temperature": 0.7
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", agent_name)
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", agent_name)
        
        return {"text_content": response.text.strip() if response.text else "Vídeo processado (planejado)."}, []

class ThinkingWorker(Worker):
    """Agente especializado em pensamento complexo e raciocínio estruturado."""
    
    def __init__(self, task_manager):
        super().__init__(task_manager)
        log_message("ThinkingWorker (v12.0 - Chain of Thought) criado.", "ThinkingWorker")
    
    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "ThinkingWorker"
        print_agent_message(agent_name, f"Executando (pensamento): '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(
            f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
            f"Objetivo Geral: {original_goal}\n\n"
            f"TAREFA DE PENSAMENTO COMPLEXO: {task_description}\n\n"
            f"Instruções especiais:\n"
            f"1. Use raciocínio passo-a-passo (chain-of-thought)\n"
            f"2. Comece explicando seu PENSAMENTO: antes de dar a resposta\n"
            f"3. Considere múltiplas perspectivas e alternativas\n"
            f"4. Identifique possíveis problemas ou limitações\n"
            f"5. Forneça justificativas para suas conclusões\n"
            f"6. Se necessário, use ferramentas para salvar análises detalhadas\n\n"
            f"Formato preferido:\n"
            f"PENSAMENTO: [seu raciocínio detalhado aqui]\n"
            f"RESPOSTA: [sua conclusão/solução aqui]"
        )
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS,
            "temperature": 0.4,  # Equilibrio entre criatividade e precisão
            "max_output_tokens": 2048  # Permite respostas mais elaboradas
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", agent_name)
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", agent_name)
        
        return {"text_content": response.text.strip() if response.text else "Pensamento estruturado concluído."}, []

class BrowserWorker(Worker):
    """Agente especializado em tarefas relacionadas a navegação web, busca no Google e automação de browser."""
    
    def __init__(self, task_manager):
        super().__init__(task_manager)
        log_message("BrowserWorker (v12.0 - Google Search + Browser Automation) criado.", "BrowserWorker")
    
    def execute_task(self, task_description, previous_results, files_info, original_goal):
        agent_name = "BrowserWorker"
        print_agent_message(agent_name, f"Executando (web/browser): '{task_description}'")

        conversation_history = []
        if self.task_manager.uploaded_file_objects:
             conversation_history.extend(self.task_manager.uploaded_file_objects)
        
        conversation_history.append(
            f"Contexto: {json.dumps(previous_results) if previous_results else 'Nenhum.'}\n"
            f"Objetivo Geral: {original_goal}\n\n"
            f"TAREFA DE NAVEGAÇÃO/PESQUISA WEB: {task_description}\n\n"
            f"Ferramentas disponíveis para você:\n"
            f"• google_search: Para buscar informações no Google\n"
            f"• fetch_webpage_content: Para extrair conteúdo de páginas web\n"
            f"• browser_automation: Para navegar, buscar texto e extrair links\n\n"
            f"Use essas ferramentas conforme necessário para completar a tarefa web."
        )
        
        gen_config = {
            "tools": AVAILABLE_TOOL_DECLARATIONS,
            "temperature": 0.4  # Equilibrio para pesquisa efetiva
        }

        response = call_gemini_api_with_retry(conversation_history, agent_name, gen_config_dict=gen_config)

        if not response: return {"text_content": "Falha na API."}, []

        extract_and_print_thoughts(response)
        
        # Handle function calls if any
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in AVAILABLE_TOOLS:
                        log_message(f"Executando função: {function_name} com args: {function_args}", agent_name)
                        result = AVAILABLE_TOOLS[function_name](**function_args)
                        log_message(f"Resultado da função {function_name}: {result}", agent_name)
        
        return {"text_content": response.text.strip() if response.text else "Tarefa web concluída."}, []

class TaskManager:
    def __init__(self, initial_goal, uploaded_files, files_meta):
        self.goal = initial_goal
        self.uploaded_file_objects = uploaded_files or []
        self.uploaded_files_info = files_meta or []
        self.executed_tasks_results = []
        
        # Initialize router and specialized workers
        self.router = RouterAgent()
        self.text_worker = Worker(self)
        self.image_worker = ImageWorker(self)
        self.analysis_worker = AnalysisWorker(self)
        self.video_worker = VideoWorker(self)
        self.thinking_worker = ThinkingWorker(self)
        self.browser_worker = BrowserWorker(self)
        
        # Map agent types to worker instances
        self.worker_map = {
            "text_worker": self.text_worker,
            "image_worker": self.image_worker,
            "analysis_worker": self.analysis_worker,
            "video_worker": self.video_worker,
            "thinking_worker": self.thinking_worker,
            "browser_worker": self.browser_worker
        }
        
        self.system_instruction = (
            "Você é um Gerenciador de Tarefas especialista. Decomponha a meta principal em sub-tarefas sequenciais e executáveis. "
            "Sua resposta DEVE ser um objeto JSON bem formado contendo uma única chave 'tasks', que é uma lista de strings. Exemplo: {\\'tasks\\': [\\'Passo 1\\', \\'Passo 2\\']}"
        )
        log_message("TaskManager (v12.0 - Gemini 2.5 com Router + Google Search + Browser Tools) criado.", "TaskManager")
        
    def decompose_goal(self):
        agent_name = "Task Manager"
        print_agent_message(agent_name, f"Decompondo meta: '{self.goal}'")

        prompt_text = (f"{self.system_instruction}\n\nMeta a ser decomposta: \\'{self.goal}\\'")
        prompt_parts = []
        if self.uploaded_file_objects: prompt_parts.extend(self.uploaded_file_objects)
        prompt_parts.append(prompt_text)
        
        gen_config = {
            "temperature": 0.5, 
            "response_mime_type": "application/json"
        }
        response = call_gemini_api_with_retry(prompt_parts, agent_name, gen_config_dict=gen_config)
        
        if not response or not response.text: return [self.goal]
        
        extract_and_print_thoughts(response)
        try:
            text_response = response.text.strip()
            if text_response.startswith("```json"): text_response = text_response[7:-3].strip()
            plan_dict = json.loads(text_response)
            tasks = plan_dict.get("tasks", [])
            return tasks if isinstance(tasks, list) else [self.goal]
        except (json.JSONDecodeError, TypeError) as e:
            log_message(f"Falha ao decodificar JSON do planejador: {e}. Resposta: '{response.text}'", "TaskManager")
            return [self.goal]
    
    def run_workflow(self):
        print_agent_message("TaskManager", "Iniciando fluxo de trabalho...")
        task_list = self.decompose_goal()
        
        print_agent_message("TaskManager", "--- PLANO DE TAREFAS ---")
        for i, task in enumerate(task_list): print(f"  {i+1}. {task}")
        
        if input("👤 Aprova? (s/n) ➡️ ").strip().lower() != 's':
            print_agent_message("TaskManager", "Plano não aprovado."); return
        
        for task in task_list:
            # Use router to determine best worker for this task
            context = json.dumps(self.executed_tasks_results) if self.executed_tasks_results else ""
            agent_type, reasoning = self.router.route_task(task, context)
            
            # Get the appropriate worker
            worker = self.worker_map.get(agent_type, self.text_worker)
            
            print_agent_message("TaskManager", f"Executando '{task}' com {agent_type}")
            
            result, _ = worker.execute_task(
                task, self.executed_tasks_results, 
                self.uploaded_files_info, self.goal
            )
            self.executed_tasks_results.append({task: result})
            print_agent_message("TaskManager", f"Resultado da tarefa '{task}': {result.get('text_content')}")

        print_agent_message("TaskManager", "Fluxo de trabalho concluído!")

# --- Função Principal ---
if __name__ == "__main__":
    SCRIPT_VERSION = "v12.0 (Gemini 2.5 Preview + RouterAgent)"
    log_message(f"--- Início ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"--- Sistema Multiagente Gemini ({SCRIPT_VERSION}) ---")
    
    files, meta = get_uploaded_files_info_from_user()
    
    print_user_message("🎯 Defina a meta principal (digite 'FIM' para concluir):")
    initial_goal = "\n".join(iter(input, 'FIM'))

    if not initial_goal.strip():
        print("Nenhuma meta definida.")
    else:
        log_message(f"Meta: {initial_goal}", "Usuário")
        TaskManager(initial_goal, files, meta).run_workflow()

    log_message(f"--- Fim ({SCRIPT_VERSION}) ---", "Sistema")
    print(f"\\n--- Execução ({SCRIPT_VERSION}) Finalizada ---")

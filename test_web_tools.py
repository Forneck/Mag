#!/usr/bin/env python3
"""
Teste simples para verificar se as novas ferramentas web funcionam corretamente
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import urllib.parse

def google_search(query: str, num_results: int = 5) -> dict:
    """Realiza uma busca no Google e retorna os resultados com títulos e links."""
    try:
        print(f"Buscando no Google: '{query}' (máximo {num_results} resultados)")
        
        search_results = []
        for result in search(query, stop=num_results, pause=2):
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
        
        return {
            "status": "success", 
            "message": result_text,
            "results": detailed_results,
            "count": len(detailed_results)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Erro ao buscar no Google: {e}"}

def fetch_webpage_content(url: str, extract_text_only: bool = True) -> dict:
    """Busca o conteúdo de uma página web e extrai texto ou HTML."""
    try:
        print(f"Buscando conteúdo da página: {url}")
        
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
        return {"status": "error", "message": f"Erro ao acessar a página: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar página: {e}"}

def test_google_search():
    """Testa a função google_search"""
    print("=== Testando google_search ===")
    try:
        result = google_search("Python programming", 3)
        print(f"Status: {result.get('status')}")
        print(f"Número de resultados: {result.get('count', 0)}")
        if result.get('status') == 'success':
            print("✅ google_search funcionando corretamente")
        else:
            print(f"❌ Erro: {result.get('message')}")
    except Exception as e:
        print(f"❌ Erro ao testar google_search: {e}")

def test_fetch_webpage():
    """Testa a função fetch_webpage_content"""
    print("\n=== Testando fetch_webpage_content ===")
    try:
        result = fetch_webpage_content("https://httpbin.org/html")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"Título encontrado: {result.get('title', 'N/A')}")
            content_length = len(result.get('content', ''))
            print(f"Conteúdo extraído: {content_length} caracteres")
            print("✅ fetch_webpage_content funcionando corretamente")
        else:
            print(f"❌ Erro: {result.get('message')}")
    except Exception as e:
        print(f"❌ Erro ao testar fetch_webpage_content: {e}")

def test_imports():
    """Testa se todas as importações funcionam"""
    print("=== Testando importações ===")
    try:
        import requests
        print("✅ requests importado")
        
        from bs4 import BeautifulSoup
        print("✅ BeautifulSoup importado")
        
        from googlesearch import search
        print("✅ googlesearch importado")
        
        print("✅ Todas as importações funcionando")
        return True
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

if __name__ == "__main__":
    print("Testando novas ferramentas web do MAG...")
    
    if not test_imports():
        print("❌ Falha nas importações. Verifique se todas as dependências estão instaladas.")
        sys.exit(1)
    
    test_google_search()
    test_fetch_webpage()
    
    print("\n=== Testes concluídos ===")
    print("Nota: Alguns testes podem falhar devido a limitações de rede ou rate limiting.")
#!/usr/bin/env python3
"""
Exemplo de uso das novas funcionalidades web do MAG
Demonstra como usar Google Search e navegação web sem precisar de API keys
"""

# Exemplo de como as novas ferramentas podem ser usadas:

def example_usage():
    print("=== Exemplo de Uso das Novas Ferramentas Web do MAG ===\n")
    
    print("1. GOOGLE SEARCH:")
    print("   Função: google_search(query, num_results=5)")
    print("   Exemplo: google_search('Python machine learning tutorial', 3)")
    print("   Retorna: Lista de resultados com títulos e URLs\n")
    
    print("2. NAVEGAÇÃO WEB:")
    print("   Função: fetch_webpage_content(url, extract_text_only=True)")
    print("   Exemplo: fetch_webpage_content('https://python.org')")
    print("   Retorna: Conteúdo de texto limpo da página\n")
    
    print("3. AUTOMAÇÃO DE BROWSER:")
    print("   Função: browser_automation(action, url, element_selector='', ...)")
    print("   Ações disponíveis:")
    print("   - 'navigate': Navega para uma URL")
    print("   - 'search_content': Busca texto específico na página")
    print("   - 'extract_links': Extrai todos os links da página")
    print("   Exemplo: browser_automation('extract_links', 'https://news.ycombinator.com')\n")
    
    print("4. ROTEAMENTO INTELIGENTE:")
    print("   O RouterAgent agora reconhece tarefas relacionadas a web e")
    print("   automaticamente direciona para o BrowserWorker que tem acesso")
    print("   a todas essas ferramentas.\n")
    
    print("=== Exemplos de Tarefas que Acionam o BrowserWorker ===")
    examples = [
        "Busque informações sobre inteligência artificial no Google",
        "Visite o site da OpenAI e extraia o conteúdo principal",
        "Procure tutoriais de Python e me dê uma lista de links",
        "Pesquise notícias sobre tecnologia e resuma os principais tópicos",
        "Navegue no site do GitHub e encontre projetos relacionados a ML"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\n=== Ferramentas Disponíveis no BrowserWorker ===")
    tools = [
        "google_search: Busca no Google com resultados estruturados",
        "fetch_webpage_content: Extração de conteúdo web limpo",
        "browser_automation: Navegação e extração de links",
        "save_file: Salvar resultados em arquivos"
    ]
    
    for tool in tools:
        print(f"• {tool}")
    
    print("\n=== Fluxo de Trabalho Típico ===")
    workflow = [
        "1. Usuário define meta: 'Pesquise sobre Python e crie um relatório'",
        "2. TaskManager decompõe em subtarefas",
        "3. RouterAgent identifica tarefas web e direciona para BrowserWorker",
        "4. BrowserWorker usa google_search para encontrar informações",
        "5. BrowserWorker usa fetch_webpage_content para extrair conteúdo",
        "6. BrowserWorker usa save_file para criar o relatório final"
    ]
    
    for step in workflow:
        print(step)
    
    print("\n=== Configuração Necessária ===")
    print("Apenas instale as dependências:")
    print("pip install googlesearch-python beautifulsoup4 requests")
    print("\nNenhuma API key adicional necessária para as ferramentas web!")

if __name__ == "__main__":
    example_usage()
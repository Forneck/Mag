# MAG: Sistema Multiagente Gemini (v12.0 - Gemini 2.5 Preview + Google Search + Browser Tools)

## Descrição

MAG (Multi-Agent Gemini) é um sistema baseado em Python que utiliza os Modelos de Linguagem Grandes (LLMs) Gemini 2.5 Preview do Google para automatizar tarefas complexas através de uma arquitetura multiagente inteligente. O sistema agora inclui um **RouterAgent** que automaticamente seleciona o agente especializado mais adequado para cada tarefa, oferecendo capacidades aprimoradas de raciocínio, geração de imagens, navegação web, busca no Google e automação de browser.

## Funcionalidades

* **Roteamento Inteligente**: Novo **RouterAgent** que analisa cada tarefa e automaticamente seleciona o agente especializado mais adequado.
* **Decomposição de Tarefas**: O **TaskManager** divide metas complexas em uma lista sequencial de subtarefas usando o Gemini 2.5 Preview.
* **Agentes Especializados**:
    * **Worker**: Executa subtarefas gerais baseadas em texto e código.
    * **ImageWorker**: Dedicado a gerar e processar imagens usando Gemini 2.0 Flash.
    * **AnalysisWorker**: Especializado em análise de dados e processamento complexo.
    * **ThinkingWorker**: Focado em raciocínio estruturado, chain-of-thought e resolução de problemas complexos.
    * **VideoWorker**: Preparado para geração de vídeos com Veo3 (quando disponível na API).
    * **BrowserWorker**: 🆕 Especializado em navegação web, busca no Google e automação de browser.
* **Ferramentas Web Avançadas**: 🆕
    * **Google Search**: Busca inteligente no Google com extração de títulos e snippets
    * **Navegação Web**: Extração de conteúdo limpo de páginas web
    * **Automação de Browser**: Navegação, busca de texto e extração de links
* **Capacidades Aprimoradas de Pensamento**: Suporte melhorado para chain-of-thought reasoning e análise estruturada.
* **Ferramenta de Geração de Imagens**: Integração completa com modelos de geração de imagens do Gemini.
* **Preparação para Veo3**: Framework pronto para integração com geração de vídeos quando a API estiver disponível.
* **Upload de Arquivos e Contexto**: Suporta o upload de arquivos locais (com wildcards, ex: `*.txt`) para fornecer contexto adicional aos modelos Gemini.
* **Gerenciamento de Cache**: Permite ao usuário visualizar e, opcionalmente, limpar o cache de uploads local e os arquivos na API Gemini antes de iniciar uma nova sessão.
* **Ciclo de Validação e Feedback Robusto**:
    * Ao final do fluxo de tarefas, o Validator avalia o resultado.
    * O usuário recebe um menu claro (`[A]provar`, `[F]eedback`, `[S]air`) para aprovar os artefatos, fornecer feedback para uma nova iteração ou encerrar o processo, evitando loops indesejados.
* **Logs Abrangentes**: Registro detalhado de todas as operações, mensagens dos agentes e chamadas de API para rastreabilidade e depuração.
* **Gerenciamento de Artefatos**: Utiliza um diretório temporário para os artefatos gerados, que são movidos para a pasta de saída final apenas após a aprovação.

## Arquitetura

O sistema é construído sobre um padrão Gerenciador/Router/Trabalhadores Especializados:

* **TaskManager**: Atua como o orquestrador central. Ele interpreta o objetivo, o divide em um plano, e delega as subtarefas aos agentes apropriados.
* **RouterAgent**: Agente inteligente que analisa cada tarefa e determina qual agente especializado é mais adequado para executá-la.
* **Worker**: Agente de propósito geral responsável por executar subtarefas baseadas em texto e código.
* **ImageWorker**: Agente especializado focado em gerar e processar imagens.
* **AnalysisWorker**: Agente especializado em análise de dados e processamento analítico.
* **ThinkingWorker**: Agente focado em raciocínio complexo, chain-of-thought e resolução de problemas estruturados.
* **VideoWorker**: Agente preparado para geração de vídeos (Veo3 quando disponível).
* **BrowserWorker**: 🆕 Agente especializado em navegação web, busca no Google e automação de browser.

## Componentes/Classes Chave

* **RouterAgent**:
    * `route_task()`: Analisa tarefas e seleciona o agente especializado mais adequado.
* **TaskManager**:
    * `run_workflow()`: Gerencia o fluxo de execução geral, incluindo roteamento inteligente de tarefas.
    * `decompose_goal()`: Decompõe a meta principal usando Gemini 2.5 Preview.
* **Worker**: Agente geral para tarefas de texto e código.
* **ImageWorker**: Agente especializado para geração e processamento de imagens.
* **AnalysisWorker**: Agente especializado para análise de dados.
* **ThinkingWorker**: Agente especializado em raciocínio complexo e chain-of-thought.
* **VideoWorker**: Agente preparado para geração de vídeos (Veo3).
* **BrowserWorker**: 🆕 Agente especializado para navegação web e busca.
* **Funções de Ferramentas**:
    * `save_file()`: Salva conteúdo em arquivos.
    * `generate_image()`: Gera imagens usando Gemini 2.0 Flash.
    * `generate_video()`: Planeja geração de vídeos (Veo3 quando disponível).
    * `google_search()`: 🆕 Realiza buscas no Google com resultados estruturados.
    * `fetch_webpage_content()`: 🆕 Extrai conteúdo limpo de páginas web.
    * `browser_automation()`: 🆕 Automação básica de navegação e extração de links.

## Como Funciona (Fluxo de Trabalho)

1.  **Gerenciamento de Arquivos**: O usuário pode gerenciar arquivos existentes na API e fazer upload de novos arquivos.
2.  **Entrada do Usuário**: O usuário fornece uma meta principal.
3.  **Decomposição e Aprovação**: O TaskManager cria um plano de tarefas usando Gemini 2.5 Preview, que o usuário aprova.
4.  **Loop de Execução com Roteamento Inteligente**:
    * O TaskManager itera pela lista de tarefas.
    * Para cada tarefa, o **RouterAgent** analisa o conteúdo e seleciona automaticamente o agente especializado mais adequado.
    * O agente selecionado (Worker, ImageWorker, AnalysisWorker, ThinkingWorker, VideoWorker, ou BrowserWorker) executa a tarefa.
    * Resultados são coletados e contextualizados para as próximas tarefas.
5.  **Processamento Especializado**:
    * **ImageWorker** gera imagens usando Gemini 2.0 Flash.
    * **ThinkingWorker** aplica raciocínio estruturado e chain-of-thought.
    * **AnalysisWorker** processa dados analíticos.
    * **VideoWorker** planeja geração de vídeos para implementação futura com Veo3.
    * **BrowserWorker** 🆕 executa buscas no Google, navega em sites e extrai conteúdo web.

## Configuração/Pré-requisitos

* Python 3.8+
* Chave da API Google Gemini 2.5 Preview
* Pacotes Python necessários:
    ```bash
    pip install google-generativeai pillow googlesearch-python beautifulsoup4 requests
    ```

## Variáveis de Ambiente

* Defina a variável de ambiente `GEMINI_API_KEY` com sua chave da API:
    ```bash
    export GEMINI_API_KEY="SUA_CHAVE_API"
    ```

## Uso

1.  Certifique-se de que você tem o Python 3.8+ instalado e a variável de ambiente `GEMINI_API_KEY` configurada.
2.  Execute o script: `python mag.py`
3.  O sistema irá guiá-lo através das etapas: gerenciamento de arquivos, definição da meta e aprovação do plano.
4.  O RouterAgent automaticamente selecionará os melhores agentes especializados para cada tarefa.

## Configuração

Vários parâmetros podem ser configurados no início do script `mag.py`:

* **Diretórios**: `LOG_DIRECTORY`, `OUTPUT_DIRECTORY`.
* **Retentativas**: `MAX_API_RETRIES`.
* **Modelos**: `GEMINI_TEXT_MODEL_NAME` (Gemini 2.5 Preview), `GEMINI_IMAGE_MODEL_NAME` (Gemini 2.0 Flash).

## Novidades da Versão 12.0 (Gemini 2.5 Preview + Web Tools)

* **Compatibilidade com Gemini 2.5 Preview**: Atualização completa da API.
* **RouterAgent**: Sistema inteligente de roteamento automático de tarefas.
* **Agentes Especializados**: ImageWorker, AnalysisWorker, ThinkingWorker, VideoWorker, BrowserWorker.
* **Ferramentas Web**: 🆕 Google Search, navegação web e automação de browser.
* **BrowserWorker**: 🆕 Agente especializado para tarefas relacionadas à web.
* **Capacidades de Pensamento Aprimoradas**: Melhor suporte para chain-of-thought reasoning.
* **Preparação para Veo3**: Framework pronto para geração de vídeos quando disponível.
* **Ferramenta de Vídeo**: Planejamento e documentação para futura integração com Veo3.

## Estrutura de Arquivos (Saídas)

* `gemini_agent_logs/`: Contém logs detalhados de cada execução.
* `gemini_uploaded_files_cache/`: Armazena metadados de arquivos carregados.
* `gemini_temp_artifacts/`: **(Novo)** Armazena temporariamente os artefatos gerados durante a execução (imagens, código). É limpo no início e no fim.
* `gemini_final_outputs/`:
    * Contém subdiretórios com timestamp para cada execução bem-sucedida.
    * Dentro de cada subdiretório, armazena os **artefatos finais aprovados** e o **relatório de avaliação** em Markdown.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para enviar um Pull Request ou abrir uma Issue.

## Licença

Este projeto está licenciado sob a Licença MIT.

# MAG: Sistema Multiagente Gemini (v9.4)

## Descrição

MAG (Multi-Agent Gemini) é um sistema baseado em Python que utiliza os Modelos de Linguagem Grandes (LLMs) Gemini do Google para automatizar tarefas complexas através de uma arquitetura multiagente. Ele emprega um Agente Gerenciador de Tarefas (**TaskManager**) para decompor objetivos de alto nível em subtarefas gerenciáveis. Essas tarefas são executadas por Agentes Trabalhadores (**Worker**) especializados e, em seguida, avaliadas por um Agente Validador (**Validator**) que gerencia a qualidade e o ciclo de feedback com o usuário antes de salvar os artefatos finais.

## Funcionalidades

* **Decomposição de Tarefas**: O **TaskManager** divide metas complexas em uma lista sequencial de subtarefas usando o Gemini.
* **Agentes Especializados**:
    * **Worker**: Executa subtarefas gerais baseadas em texto e código.
    * **ImageWorker**: Dedicado a gerar imagens, salvando-as em um diretório temporário para avaliação.
    * **Validator**: Agente de qualidade que avalia os artefatos gerados (texto, código e imagens), gerencia o ciclo de feedback com o usuário e consolida a saída final.
* **Gerenciamento Dinâmico de Tarefas**: O Worker pode sugerir novas tarefas durante a execução, que o TaskManager valida e integra ao fluxo de trabalho.
* **Upload de Arquivos e Contexto**: Suporta o upload de arquivos locais (com wildcards, ex: `*.txt`) para fornecer contexto adicional aos modelos Gemini.
* **Gerenciamento de Cache**: Permite ao usuário visualizar e, opcionalmente, limpar o cache de uploads local e os arquivos na API Gemini antes de iniciar uma nova sessão.
* **Ciclo de Validação e Feedback Robusto**:
    * Ao final do fluxo de tarefas, o Validator avalia o resultado.
    * O usuário recebe um menu claro (`[A]provar`, `[F]eedback`, `[S]air`) para aprovar os artefatos, fornecer feedback para uma nova iteração ou encerrar o processo, evitando loops indesejados.
* **Logs Abrangentes**: Registro detalhado de todas as operações, mensagens dos agentes e chamadas de API para rastreabilidade e depuração.
* **Gerenciamento de Artefatos**: Utiliza um diretório temporário para os artefatos gerados, que são movidos para a pasta de saída final apenas após a aprovação.

## Arquitetura

O sistema é construído sobre um padrão Gerenciador/Trabalhador/Validador:

* **TaskManager**: Atua como o orquestrador central. Ele interpreta o objetivo, o divide em um plano, delega as subtarefas, gerencia o fluxo de informações e orquestra o ciclo de validação e feedback.
* **Worker**: Agente de propósito geral responsável por executar subtarefas baseadas em texto e código.
* **ImageWorker**: Agente especializado focado em gerar imagens com base em prompts e salvar os resultados temporariamente.
* **Validator**: Agente de "Quality Assurance". Ele avalia os conceitos de imagem gerados e realiza a validação final de todos os artefatos, interagindo com o usuário para aprovação ou iteração.

## Componentes/Classes Chave

* **TaskManager**:
    * `run_workflow()`: Gerencia o fluxo de execução geral, incluindo o novo ciclo de validação e feedback.
    * `decompose_task()`: Decompõe a meta principal.
    * `confirm_new_tasks_with_llm()`: Valida novas tarefas sugeridas.
* **Worker**:
    * `execute_sub_task()`: Executa uma subtarefa e pode sugerir novas.
* **ImageWorker**:
    * `generate_image()`: Gera uma imagem e a salva em um diretório temporário, retornando o caminho do arquivo.
* **Validator**:
    * `evaluate_and_select_image_concepts()`: Seleciona os prompts de imagem mais promissores para aprovação.
    * `validate_and_save_final_output()`: Realiza a validação final, gera o relatório e, após a aprovação, move os artefatos para a pasta de saída final.
* **Funções de Apoio**:
    * `clear_upload_cache()`: Permite a limpeza do cache local e da API.
    * `get_user_feedback_or_approval()`: Garante uma captura de entrada robusta do usuário para o ciclo de feedback.

## Como Funciona (Fluxo de Trabalho)

1.  **Limpeza de Cache (Opcional)**: O usuário decide se quer limpar os caches locais e da API.
2.  **Entrada do Usuário**: O usuário fornece uma meta principal e, opcionalmente, faz upload de arquivos.
3.  **Decomposição e Aprovação**: O TaskManager cria um plano de tarefas, que o usuário aprova.
4.  **Loop de Execução de Tarefas**:
    * O TaskManager itera pela lista, delegando tarefas aos Workers apropriados.
    * O `ImageWorker` gera imagens e as salva em `gemini_temp_artifacts/`.
    * Resultados e artefatos são coletados.
5.  **Ciclo de Validação e Feedback**:
    * Ao final do ciclo de tarefas, o **Validator** avalia todos os artefatos gerados.
    * Se a validação falhar, o usuário é apresentado com o menu: `[A]provar`, `[F]eedback`, `[S]air`.
    * **Feedback**: Se o usuário fornecer feedback, uma nova tarefa de correção é adicionada à lista, e o loop de execução recomeça.
    * **Aprovar/Sair**: Se o usuário aprovar ou se a validação inicial for bem-sucedida, o Validator move os artefatos aprovados do diretório temporário para a pasta de saída final (`gemini_final_outputs/`) e o processo é concluído.

## Configuração/Pré-requisitos

* Python 3.x
* Chave da API Google Gemini
* Pacotes Python necessários:
    ```bash
    pip install google-generativeai
    ```
    *(O script usa bibliotecas padrão como `os`, `json`, `shutil`, etc.)*

## Variáveis de Ambiente

* Defina a variável de ambiente `GEMINI_API_KEY` com sua chave da API:
    ```bash
    export GEMINI_API_KEY="SUA_CHAVE_API"
    ```

## Uso

1.  Certifique-se de que você tem o Python instalado e a variável de ambiente `GEMINI_API_KEY` configurada.
2.  Salve o script como `mag.py`.
3.  Execute o script a partir do seu terminal: `python mag.py`
4.  O script irá guiá-lo através das etapas: limpeza de cache (opcional), definição da meta, upload de arquivos e aprovação do plano.

## Configuração

Vários parâmetros podem ser configurados diretamente no início do script `mag.py`:

* **Diretórios**: `LOG_DIRECTORY`, `OUTPUT_DIRECTORY`, `UPLOADED_FILES_CACHE_DIR`, `TEMP_ARTIFACTS_DIR`.
* **Retentativas**: `MAX_API_RETRIES`, `MAX_AUTOMATIC_VALIDATION_RETRIES`, `MAX_MANUAL_VALIDATION_RETRIES`.
* **Modelos**: `GEMINI_TEXT_MODEL_NAME`, `GEMINI_IMAGE_GENERATION_MODEL_NAME`.
* **Configurações de Geração**: `generation_config_text`, `generation_config_image_sdk`.

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

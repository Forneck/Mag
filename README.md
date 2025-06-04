MAG: Sistema Multiagente Gemini (v9.3.3)

Descrição
MAG (Multi-Agent Gemini) é um sistema baseado em Python que utiliza os Modelos de Linguagem Grandes (LLMs) Gemini do Google para automatizar tarefas complexas através de uma arquitetura multiagente. Ele emprega um Agente Gerenciador de Tarefas (TaskManager) para decompor objetivos de alto nível do usuário em subtarefas gerenciáveis, que são então executadas por Agentes Trabalhadores (Worker) especializados, incluindo um ImageWorker para geração de imagens. O sistema suporta a geração dinâmica de tarefas, upload de arquivos para fornecer contexto e um sistema robusto de logs.

Funcionalidades
 * Decomposição de Tarefas: Um agente TaskManager divide metas complexas do usuário em uma lista sequencial de subtarefas usando o Gemini.
 * Agentes Trabalhadores Especializados:
   * Worker: Executa subtarefas gerais baseadas em texto.
   * ImageWorker: Dedicado a gerar imagens com base em descrições textuais, utilizando as capacidades de geração de imagem do Gemini.
 * Gerenciamento Dinâmico de Tarefas: O Worker pode sugerir novas tarefas durante a execução, que o TaskManager valida e integra ao fluxo de trabalho.
 * Upload de Arquivos e Contexto: Suporta o upload de arquivos locais para fornecer contexto adicional aos modelos Gemini.
 * Interação com API com Retentativas: Comunicação robusta com a API Gemini, incluindo backoff exponencial para retentativas.
 * Logs Abrangentes: Registro detalhado de todas as operações, mensagens dos agentes e chamadas de API para rastreabilidade e depuração.
 * Saída Estruturada: Gera um produto final (texto ou imagem) e um relatório de avaliação detalhado.
 * Aprovação do Usuário: Inclui uma etapa para aprovação do usuário do plano de tarefas inicial.
 * Avaliação de Conceitos de Imagem: Para tarefas envolvendo múltiplas gerações de imagem, uma etapa de avaliação baseada em LLM seleciona o conceito de imagem mais promissor.

Arquitetura
O sistema é construído sobre um padrão Gerenciador de Tarefas/Trabalhador (Task Manager/Worker):
 * TaskManager: Atua como o orquestrador central. Ele interpreta o objetivo principal do usuário, o divide em um plano (lista de subtarefas), delega essas subtarefas aos agentes trabalhadores apropriados, gerencia o fluxo de informações e contexto, e valida a saída final.
 * Worker: Um agente de propósito geral responsável por executar subtarefas baseadas em texto atribuídas pelo TaskManager. Ele também pode sugerir novas tarefas se identificar uma necessidade durante a execução.
 * ImageWorker: Um agente especializado focado unicamente na geração de imagens com base em prompts fornecidos através do fluxo de trabalho da tarefa.

Componentes/Classes Chave
 * TaskManager:
   * decompose_task(): Decompõe a meta principal em subtarefas.
   * run_workflow(): Gerencia o fluxo de execução geral das tarefas.
   * confirm_new_tasks_with_llm(): Valida novas tarefas sugeridas pelo Worker.
   * evaluate_and_select_image_concept(): Seleciona o melhor prompt/resultado de imagem quando múltiplas imagens estão envolvidas.
   * validate_and_save_final_output(): Realiza a validação final e salva o resultado.
 * Worker:
   * execute_sub_task(): Executa uma subtarefa fornecida e pode sugerir novas tarefas.
 * ImageWorker:
   * generate_image(): Gera uma imagem com base em um prompt textual.

Como Funciona (Fluxo de Trabalho)
 * Entrada do Usuário: O usuário fornece uma meta principal e, opcionalmente, faz upload de arquivos complementares.
 * Decomposição da Tarefa: O TaskManager usa o Gemini para decompor a meta principal em uma lista de subtarefas.
 * Aprovação do Usuário: O plano de tarefas inicial é apresentado ao usuário para aprovação.
 * Loop de Execução de Tarefas:
   * O TaskManager itera pela lista de tarefas.
   * Para tarefas gerais, ele delega ao Worker.
   * Para tarefas de geração de imagem (TASK_GERAR_IMAGEM:), ele usa o ImageWorker. O prompt para geração de imagem é tipicamente o resultado de uma tarefa anterior.
   * Para tarefas de avaliação de imagem (TASK_AVALIAR_IMAGENS:), o TaskManager usa o Gemini para selecionar o melhor conceito de imagem com base em descrições textuais das tentativas.
   * O Worker pode sugerir novas tarefas. Estas são validadas pelo TaskManager (usando o Gemini) e, se aprovadas, inseridas na lista de tarefas.
   * Os resultados de cada tarefa são armazenados e podem ser usados como contexto para tarefas subsequentes.
 * Validação Final e Saída: Uma vez que todas as tarefas são concluídas, o TaskManager usa o Gemini para realizar uma validação final do resultado geral. Ele então salva o produto final (por exemplo, arquivo de texto, script Python, imagem) e um log de avaliação detalhado.

Configuração/Pré-requisitos
 * Python 3.x
 * Chave da API Google Gemini
 * Pacotes Python necessários:bash
   pip install google-generativeai
   *(O script usa `os`, `json`, `time`, `datetime`, `re`, `traceback`, `base64` que são bibliotecas padrão do Python.)*

Variáveis de Ambiente
 * Defina a variável de ambiente GEMINI_API_KEY com sua chave da API Google Gemini:
   export GEMINI_API_KEY="SUA_CHAVE_API"

   Ou configure-a nas configurações de variáveis de ambiente do seu sistema.
Uso
 * Certifique-se de que você tem o Python instalado e a variável de ambiente GEMINI_API_KEY configurada.
 * Salve o script como mag.py.
 * Execute o script a partir do seu terminal:
   python mag.py

 * O script irá solicitar que você:
   * Defina a meta principal.
   * Opcionalmente, adicione arquivos complementares (forneça os caminhos locais um por um, digite 'fim' para terminar).
   * Aprove o plano de tarefas inicial gerado pelo TaskManager.

Configuração
Vários parâmetros podem ser configurados diretamente no início do script mag.py:
 * BASE_DIRECTORY, LOG_DIRECTORY, OUTPUT_DIRECTORY, UPLOADED_FILES_CACHE_DIR: Caminhos para armazenar logs, saídas e informações de arquivos carregados em cache.
 * MAX_API_RETRIES, INITIAL_RETRY_DELAY_SECONDS, RETRY_BACKOFF_FACTOR: Parâmetros para retentativas de chamadas de API.
 * GEMINI_TEXT_MODEL_NAME: Modelo usado para geração de texto e lógica (ex: gemini-2.0-flash).
 * GEMINI_IMAGE_GENERATION_MODEL_NAME: Modelo usado para geração de imagens (ex: gemini-2.0-flash-preview-image-generation).
 * generation_config_text, generation_config_image: Configurações de temperatura, top_p, top_k, etc., para os modelos de texto e imagem.
 * safety_settings_gemini: Configurações de segurança para chamadas à API Gemini.

Estrutura de Arquivos (Saídas)
Quando o script é executado, ele cria os seguintes diretórios (se não existirem) no mesmo local do script:
 * gemini_agent_logs/: Contém arquivos de log detalhados para cada execução (ex: agent_log_YYYYMMDD_HHMMSS.txt).
 * gemini_final_outputs/:
   * Armazena o produto final gerado pelo fluxo de trabalho (ex: produto_text_geral_minha-meta_YYYYMMDD_HHMMSS.txt, produto_imagem_png_base64_minha-meta-de-imagem_YYYYMMDD_HHMMSS.png).
   * Armazena um arquivo de avaliação abrangente para cada execução (ex: avaliacao_completa_minha-meta_YYYYMMDD_HHMMSS.txt).
 * gemini_uploaded_files_cache/: Armazena arquivos JSON com metadados sobre os arquivos carregados pelo usuário durante uma sessão (ex: uploaded_files_info_YYYYMMDD_HHMMSS.json).
Logs
 * Todos os eventos significativos, mensagens dos agentes, chamadas de API e erros são registrados em um arquivo com timestamp no diretório gemini_agent_logs.
 * A saída do console também fornece atualizações em tempo real sobre o progresso do agente.
Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para enviar um Pull Request ou abrir uma Issue.

Licença
Este projeto está licenciado sob a Licença MIT.

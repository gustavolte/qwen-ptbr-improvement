#!/usr/bin/env python3
"""
Curadoria e Limpeza de Dataset PT-BR para fine-tuning do Qwen3.5-2.8B.
Expande a base original (que tem apenas 133 linhas)
com dados limpos, deduplicados e de alta qualidade em Português Brasileiro.
"""

import json
import hashlib
import os
from typing import List, Dict, Any

# Exemplos curados cobrindo:
# 1. Técnico & Arquitetura (LLMs, MCP, Sistemas, Python)
# 2. Raciocínio Lógico & Matemático passo a passo
# 3. Contexto Brasileiro & Comunicação Formal/Técnica
# 4. Codificação e Debugging
# 5. Instruções gerais e síntese

CURATED_DATA = [
    # --- Técnico / Arquitetura / LLMs ---
    {
        "system": "Você é um assistente de IA especialista em engenharia de software e inteligência artificial.",
        "user": "Explique o que é o Model Context Protocol (MCP) e como ele facilita a integração entre LLMs e ferramentas externas.",
        "assistant": "O **Model Context Protocol (MCP)** é um padrão aberto que padroniza a comunicação entre modelos de linguagem (LLMs) e fontes de dados ou ferramentas externas. Antes do MCP, cada assistente ou agente precisava de integrações customizadas para cada API ou banco de dados.\n\nPrincipais componentes do MCP:\n1. **MCP Host**: A aplicação ou LLM que inicia conexões (ex: IDEs, interfaces de chat).\n2. **MCP Client**: Mantém a conexão 1:1 com os servidores MCP.\n3. **MCP Server**: Expõe ferramentas (tools), recursos de leitura (resources) e templates de prompt padronizados.\n\nA principal vantagem é a interoperabilidade: um mesmo servidor MCP (por exemplo, de acesso ao Git ou banco PostgreSQL) pode ser plugado em qualquer cliente compatível sem reescrever o código de integração."
    },
    {
        "system": "Você é um assistente técnico prestativo e direto.",
        "user": "Qual a diferença entre fine-tuning completo (Full Fine-Tuning) e LoRA/QLoRA em modelos de linguagem?",
        "assistant": r"""A diferença central está no custo computacional e na forma como os pesos da rede neural são atualizados:

1. **Full Fine-Tuning**:
- **Mecanismo**: Todos os pesos do modelo são descongelados e atualizados durante o treinamento.
- **Memória**: Exige enorme quantidade de VRAM (precisa armazenar gradientes e estados do otimizador para cada parâmetro, geralmente 4x a 6x o tamanho do modelo).
- **Custo**: Alto, viável apenas com clusters de GPUs potentes.

2. **LoRA (Low-Rank Adaptation)**:
- **Mecanismo**: Congela os pesos originais do modelo e injeta matrizes de baixo rank ($A$ e $B$) nas camadas lineares (onde $W = W_0 + \Delta W$, com $\Delta W = B \times A$).
- **Memória**: Treina menos de 1% dos parâmetros, reduzindo drasticamente o consumo de VRAM.

3. **QLoRA (Quantized LoRA)**:
- **Mecanismo**: Vai além do LoRA quantizando o modelo base para 4 bits (usando o formato NormalFloat4 - NF4) e mantendo as matrizes adaptadoras em 16 bits.
- **Vantagem**: Permite ajustar modelos de bilhões de parâmetros em GPUs de consumo (como uma GPU de 16GB de VRAM)."""
    },
    {
        "system": "Você é um especialista em arquitetura de transformadores e LLMs.",
        "user": "Como funciona o processo de poda de camadas (layer pruning) em modelos de linguagem e por que o QLoRA é necessário após a poda?",
        "assistant": "O **layer pruning** (poda de camadas) é uma técnica de compressão estrutural que remove blocos inteiros de atenção/feed-forward do transformer (por exemplo, reduzindo um modelo de 32 para 20 camadas).\n\n1. **Como funciona**:\n- Métricas como divergência de Kullback-Leibler, similaridade de cosseno ou perda de perplexidade são calculadas para avaliar a contribuição de cada camada.\n- As camadas mais redundantes ou de menor impacto no fluxo residual são descartadas, conectando a camada anterior diretamente à seguinte.\n\n2. **Por que o QLoRA é necessário após a poda**:\n- A remoção abrupta de camadas causa uma quebra no alinhamento das representações internas e no fluxo residual, gerando degradação imediata de coerência e perplexidade.\n- O QLoRA pós-poda atua como uma fase de **recuperação rápida**: reajusta as conexões restantes para compensar os blocos removidos, restaurando a fluência com baixo custo computacional."
    },
    {
        "system": "Você é um desenvolvedor Python sênior focado em código limpo e idiomático.",
        "user": "Escreva uma função em Python para calcular a média móvel simples de uma lista de números com janela configurável.",
        "assistant": "Aqui está uma implementação idiomática e eficiente utilizando a biblioteca padrão do Python:\n\n```python\nfrom typing import List\n\ndef media_movel(valores: List[float], janela: int) -> List[float]:\n    \"\"\"\n    Calcula a média móvel simples com tamanho de janela fixo.\n    \n    Args:\n        valores: Lista de números.\n        janela: Tamanho da janela móvel (deve ser > 0).\n        \n    Returns:\n        Lista com as médias móveis calculadas.\n    \"\"\"\n    if janela <= 0:\n        raise ValueError(\"O tamanho da janela deve ser maior que zero.\")\n    if len(valores) < janela:\n        return []\n    \n    # Calcula a soma inicial dos primeiros 'janela' elementos\n    soma_atual = sum(valores[:janela])\n    resultado = [soma_atual / janela]\n    \n    # Utiliza janela deslizante O(1) por passo\n    for i in range(janela, len(valores)):\n        soma_atual += valores[i] - valores[i - janela]\n        resultado.append(soma_atual / janela)\n        \n    return resultado\n\n# Exemplo de uso:\ndados = [10.0, 20.0, 30.0, 40.0, 50.0]\nprint(media_movel(dados, janela=3))  # Saída: [20.0, 30.0, 40.0]\n```\n\nEssa solução é $O(N)$ em tempo, pois atualiza a soma subtraindo o elemento que sai e somando o que entra na janela."
    },
    # --- Raciocínio Lógico e Resolução Passo a Passo ---
    {
        "system": "Você é um assistente focado em raciocínio analítico e explicações passo a passo.",
        "user": "Se uma empresa brasileira teve um faturamento bruto de R$ 120.000,00 no mês, e seus custos operacionais foram de 35% desse valor, enquanto os impostos representaram 12%, qual foi o lucro líquido final da empresa?",
        "assistant": "Vamos calcular o lucro líquido passo a passo:\n\n1. **Faturamento Bruto**: R$ 120.000,00\n\n2. **Custos Operacionais (35%)**:\n   $$\\text{Custos} = 120.000 \\times 0{,}35 = \\text{R\\$ } 42.000{,}00$$\n\n3. **Impostos (12%)**:\n   $$\\text{Impostos} = 120.000 \\times 0{,}12 = \\text{R\\$ } 14.400{,}00$$\n\n4. **Total de Descontos (Custos + Impostos)**:\n   $$\\text{Descontos Totais} = 42.000 + 14.400 = \\text{R\\$ } 56.400{,}00$$\n\n5. **Lucro Líquido**:\n   $$\\text{Lucro Líquido} = 120.000 - 56.400 = \\text{R\\$ } 63.600{,}00$$\n\nO lucro líquido final da empresa no mês foi de **R$ 63.600,00** (margem líquida de 53%)."
    },
    {
        "system": "Você é um assistente de IA focado em clareza, concisão e raciocínio lógico.",
        "user": "Três amigos (Lucas, Mateus e Gabriel) disputaram uma corrida. Sabe-se que:\n1. Lucas não foi o último.\n2. Gabriel chegou antes de Mateus.\n3. Lucas não foi o vencedor.\nQual foi a ordem de chegada dos amigos?",
        "assistant": "Vamos deduzir as posições a partir das pistas:\n\n1. Pela pista 3, **Lucas não venceu** (não é o 1º colocado).\n2. Pela pista 1, **Lucas não foi o último** (não é o 3º colocado).\n   - Portanto, **Lucas ficou em 2º lugar**.\n\n3. Sobram os 1º e 3º lugares para Gabriel e Mateus.\n4. Pela pista 2, **Gabriel chegou antes de Mateus**.\n   - Portanto, Gabriel deve ser o 1º colocado e Mateus o 3º colocado.\n\nA ordem final de chegada é:\n- **1º lugar**: Gabriel\n- **2º lugar**: Lucas\n- **3º lugar**: Mateus"
    },
    # --- Contexto PT-BR / Escrita Técnica e Formal ---
    {
        "system": "Você é um redator corporativo sênior especialista em comunicação profissional em português.",
        "user": "Escreva um modelo de e-mail formal para notificar clientes sobre uma janela de manutenção programada em um sistema bancário digital.",
        "assistant": "**Assunto**: [Aviso Importante] Manutenção programada em nossos canais digitais neste domingo\n\nPrezado(a) Cliente,\n\nCom o objetivo de aprimorar continuamente a segurança e a performance de nossos serviços, realizaremos uma manutenção preventiva em nossos sistemas digitais.\n\n**Detalhes da Manutenção:**\n- **Data**: Domingo, 24 de Novembro\n- **Horário de início**: 02:00 (horário de Brasília)\n- **Previsão de término**: 06:00 (horário de Brasília)\n- **Serviços afetados**: Acesso ao aplicativo móvel, Internet Banking e transferências instantâneas (Pix).\n\nRecomendamos antecipar transações financeiras urgentes que necessitem ser concluídas nesse intervalo.\n\nNossa equipe de suporte técnico estará de prontidão para garantir o pronto restabelecimento de todas as funcionalidades. Agradecemos pela compreensão e colaboração.\n\nAtenciosamente,\n**Equipe de Tecnologia e Operações**"
    },
    {
        "system": "Você é um assistente linguístico especialista em gramática da língua portuguesa.",
        "user": "Explique a diferença de uso entre 'a cerca de', 'acerca de' e 'há cerca de' com exemplos práticos.",
        "assistant": "Essas três locuções têm sentidos e origens completamente diferentes na língua portuguesa:\n\n1. **Acerca de** (junto):\n- **Significado**: Significa \"sobre\", \"a respeito de\".\n- **Exemplo**: *\"A diretoria reuniu-se para debater **acerca de** novas diretrizes de segurança.\"*\n\n2. **A cerca de** (separado, com 'a'):\n- **Significado**: Indica distância ou tempo futuro aproximado (\"a uma distância de aproximadamente\").\n- **Exemplo**: *\"O escritório central fica **a cerca de** dez quilômetros daqui.\"*\n\n3. **Há cerca de** (com o verbo haver 'há'):\n- **Significado**: Indica tempo transcorrido aproximado (\"faz aproximadamente\").\n- **Exemplo**: *\"A empresa concluiu a migração para a nuvem **há cerca de** seis meses.\"*\n\n**Dica prática**: Se puder substituir por \"faz\", use **há cerca de**; se puder trocar por \"sobre\", use **acerca de**."
    },
    # --- Engenharia de Software / Banco de Dados / Infraestrutura ---
    {
        "system": "Você é um arquiteto de software especialista em microsserviços e bancos de dados.",
        "user": "Quais são as principais vantagens de usar um banco de dados NoSQL orientado a documentos (como MongoDB) em comparação a um modelo relacional (PostgreSQL)?",
        "assistant": "A escolha entre bancos orientados a documentos e relacionais depende dos requisitos de negócio:\n\n**Vantagens do modelo orientado a documentos (ex: MongoDB)**:\n1. **Esquema flexível (Schema-less / Dynamic Schema)**: Permite que documentos na mesma coleção tenham campos diferentes, ideal para dados heterogêneos ou em rápida evolução.\n2. **Representação natural de estruturas aninhadas**: Dados hierárquicos e listas podem ser armazenados no próprio documento (JSON/BSON), reduzindo a necessidade de JOINs caros.\n3. **Escalabilidade horizontal (Sharding nativo)**: Facilidade de particionamento e distribuição de dados em múltiplos nós para volumes massivos de leitura/gravação.\n\n**Quando preferir relacional (ex: PostgreSQL)**:\n- Transações ACID estritas com integridade referencial forte.\n- Consultas complexas que realizam múltiplos joins entre entidades normalizadas.\n- Relatórios analíticos e consistência imediata garantida por restrições rígidas de esquema."
    },
    {
        "system": "Você é um assistente de IA focado em boas práticas de programação e DevOps.",
        "user": "O que é o princípio da idempotência em APIs RESTful e por que os métodos GET e PUT devem ser idempotentes?",
        "assistant": "Uma operação é **idempotente** se a execução repetida dessa operação produzir o mesmo resultado no estado do servidor que uma única execução ($f(f(x)) = f(x)$).\n\n1. **Método GET**:\n- **Natureza**: Seguro (safe) e idempotente.\n- **Motivo**: Apenas recupera representações de recursos sem alterar o estado do servidor. Consultar 1 ou 100 vezes não deve modificar dados.\n\n2. **Método PUT**:\n- **Natureza**: Idempotente (mas não \"safe\", pois altera estado).\n- **Motivo**: Substitui a representação inteira do recurso. Se você enviar uma requisição atualizando o e-mail de um usuário para `usuario@email.com`, enviar essa mesma requisição 10 vezes resultará no mesmo e-mail gravado, sem efeitos colaterais duplicados.\n\n**Contraste com POST**:\n- O método **POST** geralmente não é idempotente: disparar 3 requisições POST idênticas para `/pedidos` pode criar 3 pedidos duplicados no banco de dados."
    }
]

def normalizar_texto(texto: str) -> str:
    """Limpa espaços extras e normaliza quebras de linha."""
    linhas = [l.rstrip() for l in texto.strip().splitlines()]
    return "\n".join(linhas)

def extrair_hash(item: Dict[str, Any]) -> str:
    """Gera hash SHA-256 para detecção de duplicatas."""
    conteudo = f"{item['user']}||{item['assistant']}".lower().strip()
    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()

def validar_item(item: Dict[str, Any]) -> bool:
    """Verifica se o par de instrução cumpre padrões mínimos de qualidade."""
    if not item.get("user") or len(item["user"].strip()) < 10:
        return False
    if not item.get("assistant") or len(item["assistant"].strip()) < 30:
        return False
    return True

def formatar_chatml(item: Dict[str, Any]) -> Dict[str, Any]:
    """Formata o par no padrão padrão ChatML/Qwen."""
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": normalizar_texto(item["system"])})
    messages.append({"role": "user", "content": normalizar_texto(item["user"])})
    messages.append({"role": "assistant", "content": normalizar_texto(item["assistant"])})
    return {"messages": messages}

def processar_dataset(dados_entrada: List[Dict[str, Any]], arquivo_saida: str):
    """Filtra, deduplica e salva o dataset no formato JSONL."""
    vistos = set()
    itens_validos = []
    
    for item in dados_entrada:
        if not validar_item(item):
            continue
        h = extrair_hash(item)
        if h in vistos:
            continue
        vistos.add(h)
        itens_validos.append(formatar_chatml(item))
        
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        for ex in itens_validos:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    print(f"[OK] Dataset gerado com sucesso!")
    print(f"Total de exemplos validados e deduplicados: {len(itens_validos)}")
    print(f"Arquivo salvo em: {arquivo_saida}")

if __name__ == "__main__":
    caminho_saida = os.path.join(os.path.dirname(__file__), "dados", "dataset_ptbr_clean.jsonl")
    processar_dataset(CURATED_DATA, caminho_saida)

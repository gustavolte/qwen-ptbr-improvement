# Fine-Tuning e Avaliação do Qwen3.5-2.8B para Português Brasileiro

Projeto desenvolvido para avaliar e ajustar o modelo `Rcscan/Qwen3.5-2.8B-V0.1` (variante reduzida de 32 para 20 camadas do Qwen 3.5 4B), com foco em recuperação de vocabulário e coerência em português.

O pipeline cobre medição de perplexidade inicial, curadoria e limpeza de dados de instrução em formato ChatML, ajuste via QLoRA em 4 bits e publicação do modelo resultante.

## Estrutura do Projeto

```
qwen-ptbr-improvement/
├── curate_dataset.py                 # Script de filtragem e formatação do dataset
├── eval_perplexity.py                # Script de medição de perplexidade no corpus de teste
├── train_qlora_and_upload.py         # Treinamento com SFTTrainer/PEFT e exportação
├── upload_to_hf_api.py               # Utilitário para envio direto via API REST
├── qwen3.5_ptbr_finetune_colab.ipynb # Notebook pronto para execução em GPU (Google Colab)
├── dados/
│   └── dataset_ptbr_clean.jsonl      # Dados de treino processados
├── RELATORIO.md                      # Relatório técnico com detalhes do experimento
└── README.md
```

## Dados de Treino

O conjunto em `dados/dataset_ptbr_clean.jsonl` segue o padrão de mensagens (`system`, `user`, `assistant`) compatível com o template do Qwen.

O script `curate_dataset.py` faz a validação de tamanho mínimo, normalização de quebras de linha e remoção de duplicatas por hash SHA-256 das entradas. Os tópicos cobrem:

- Explicações de arquitetura de software, protocolos e bancos de dados;
- Raciocínio matemático e resolução de problemas passo a passo;
- Escrita formal e técnica em português;
- Algoritmos e implementação em Python.

Para gerar ou atualizar o arquivo:

```bash
python3 curate_dataset.py
```

## Hiperparâmetros do QLoRA

O ajuste fino utiliza quantização NF4 de 4 bits para viabilizar o treino em GPUs com menos de 16 GB de VRAM:

- **Rank LoRA ($r$):** 32
- **Alpha ($\alpha$):** 64
- **Dropout:** 0.05
- **Módulos adaptados:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Taxa de aprendizado:** 2e-4 (cosine decay, warmup de 5%)
- **Otimizador:** Paged AdamW 8-bit
- **Batch size efetivo:** 8 (batch 2 com 4 passos de acumulação de gradiente)
- **Épocas:** 3

## Como Executar

### 1. Via Google Colab (com GPU T4)

Para treinar sem custos de infraestrutura local:

1. Abra o arquivo `qwen3.5_ptbr_finetune_colab.ipynb` no Google Colab.
2. Defina o runtime para GPU (T4).
3. Informe seu token do Hugging Face e o nome do repositório destino na célula de configuração.
4. Execute as células em sequência. O notebook calcula a perplexidade inicial, treina o adaptador e faz o upload dos pesos finais.

### 2. Em Máquina Local com GPU

Instale as dependências:

```bash
pip install torch transformers datasets peft accelerate bitsandbytes trl
```

Avaliação de linha de base:

```bash
python3 eval_perplexity.py Rcscan/Qwen3.5-2.8B-V0.1
```

Execução do treino e upload:

```bash
python3 train_qlora_and_upload.py \
  --base_model Rcscan/Qwen3.5-2.8B-V0.1 \
  --dataset_path dados/dataset_ptbr_clean.jsonl \
  --hf_repo_id seu-usuario/qwen3.5-2.8b-ptbr \
  --hf_token SEU_TOKEN_HF \
  --epochs 3 \
  --lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64
```

## Autor

Gustavo Leite do Nascimento  
Engenharia de Computação — UFRN  
gustavoleitte12@gmail.com

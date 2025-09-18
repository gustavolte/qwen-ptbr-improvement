# Relatório Técnico: Ajuste Fino do Qwen3.5-2.8B em Português

**Autor:** Gustavo Leite do Nascimento  
**Curso:** Engenharia de Computação — UFRN  
**Contato:** gustavoleitte12@gmail.com  
**Data:** Setembro de 2025  

---

## 1. Contextualização

O modelo `Rcscan/Qwen3.5-2.8B-V0.1` foi obtido a partir de poda estrutural de camadas do Qwen 3.5 4B, reduzindo a arquitetura original de 32 para 20 blocos de atenção híbrida (atenção linear combinada com atenção padrão em intervalos de 4 camadas).

Embora a redução preserve a maior parte dos parâmetros ativos (~2.8B) e reduza a pegada de memória, a retirada de 12 blocos afeta a distribuição das representações internas e a coerência textual. A recuperação pós-poda dependeu de um conjunto de treino reduzido (133 entradas), o que deixa espaço para melhorias substanciais por meio de curadoria de dados em português e calibração de hiperparâmetros de adaptação de baixo posto (LoRA).

## 2. Metodologia

O trabalho seguiu três etapas principais:

### 2.1. Medição de Perplexidade

Para estabelecer uma métrica objetiva antes de qualquer alteração, o script `eval_perplexity.py` calcula a perda de entropia cruzada em um conjunto de sentenças técnicas em português:

$$PPL = \exp\left(\frac{1}{N} \sum_{i=1}^N \mathcal{L}_i\right)$$

Essa métrica permite verificar numericamente se as mudanças propostas reduzem a incerteza do modelo sobre o texto em português.

### 2.2. Curadoria e Filtros de Dados

O dataset foi construído e padronizado pelo script `curate_dataset.py`, aplicando:

- Validação de comprimento mínimo (evita instruções ou respostas vazias/truncadas);
- Normalização de espaçamentos e quebras de linha;
- Deduplicação determinística via hash SHA-256 do par pergunta/resposta;
- Formatação no padrão ChatML (`system`, `user`, `assistant`).

Os exemplos abordam explicações técnicas detalhadas, resolução de problemas de lógica e matemática com passos intermediários e redação formal em português.

### 2.3. Configuração do Treinamento (QLoRA)

O pipeline em `train_qlora_and_upload.py` utiliza quantização em 4 bits (NF4) com dupla quantização para carregar a base sem sobrecarregar a memória de vídeo:

| Parâmetro | Valor |
|---|---|
| Rank LoRA ($r$) | 32 |
| Alpha ($\alpha$) | 64 |
| Módulos-alvo | Todas as projeções lineares (`q, k, v, o, gate, up, down`) |
| Precisão base | 4-bit NormalFloat (NF4) |
| Precisão dos adaptadores | bfloat16 / float16 |
| Taxa de aprendizado | 2e-4 com decaimento cosseno e warmup de 5% |
| Épocas | 3 |
| Tamanho de lote efetivo | 8 (2 por dispositivo com 4 passos de acumulação) |

O aumento do rank para 32 frente a valores típicos mais conservadores (8 ou 16) visa fornecer capacidade adicional para compensar as conexões perdidas na poda de camadas.

## 3. Ambiente de Execução e Publicação

Dado que o treinamento exige VRAM dedicada (~6 GB em 4 bits), o notebook `qwen3.5_ptbr_finetune_colab.ipynb` foi estruturado para execução em instâncias gratuitas com GPU T4 (16 GB VRAM) no Google Colab.

O notebook conduz a avaliação inicial, o treinamento por SFTTrainer e o envio direto dos pesos do adaptador ao repositório configurado no Hugging Face Hub em modo privado via `push_to_hub`.

## 4. Próximos Passos

Como continuidade, duas frentes têm potencial de ganho adicional:
1. **Destilação de conhecimento:** Utilizar a versão original de 4B como professora para guiar as probabilidades do modelo reduzido;
2. **Reavaliação das camadas podadas:** Testar outros esquemas de seleção de blocos baseados na similaridade de cosseno ou perda de ativação antes da remoção definitiva.

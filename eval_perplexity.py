#!/usr/bin/env python3
"""
Script de Medição de Linha de Base (Benchmark de Perplexidade em PT-BR)
Avalia a perplexidade do modelo Rcscan/Qwen3.5-2.8B-V0.1 antes e depois
do fine-tuning para comprovar se houve ganho de fluência e domínio em PT-BR.
"""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPUS_TESTE_PTBR = [
    "A inteligência artificial generativa tem avançado rapidamente nos últimos anos, transformando setores inteiros da economia global.",
    "O processo de poda estrutural em redes neurais profundas permite diminuir a quantidade de parâmetros sem perder completamente a capacidade de representação.",
    "No Brasil, o desenvolvimento de modelos abertos de linguagem em português é fundamental para garantir a soberania tecnológica e a preservação das nuances culturais.",
    "Para calcular a perplexidade de um modelo causal, avalia-se a probabilidade atribuída aos próximos tokens através da perda de entropia cruzada.",
    "A utilização de QLoRA reduz drasticamente a necessidade de memória de vídeo durante o treinamento, permitindo que modelos de vários bilhões de parâmetros sejam ajustados em GPUs convencionais."
]

def calcular_perplexidade(model_id_or_path: str, textos=CORPUS_TESTE_PTBR, device: str = None):
    """
    Carrega o modelo e calcula a perplexidade média sobre o corpus de teste em português.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"[*] Carregando tokenizer e modelo de: {model_id_or_path} no dispositivo: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    
    # Se estiver rodando em GPU, usa 4-bit para economizar VRAM
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True
        )
        
    model.eval()
    perdas = []
    
    with torch.no_grad():
        for i, texto in enumerate(textos, 1):
            inputs = tokenizer(texto, return_tensors="pt").to(device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            ppl = math.exp(loss)
            perdas.append(loss)
            print(f"  Exemplo {i} - Perda: {loss:.4f} | Perplexidade: {ppl:.2f}")
            
    perda_media = sum(perdas) / len(perdas)
    ppl_media = math.exp(perda_media)
    print("\n" + "="*50)
    print(f"Resultado Final para {model_id_or_path}:")
    print(f"Perda Média (Cross-Entropy Loss): {perda_media:.4f}")
    print(f"Perplexidade Média (PPL PT-BR):    {ppl_media:.2f}")
    print("="*50)
    return ppl_media

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Rcscan/Qwen3.5-2.8B-V0.1"
    calcular_perplexidade(model_name)

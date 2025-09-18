#!/usr/bin/env python3
"""
Pipeline de Fine-Tuning QLoRA e Upload Privado no Hugging Face
Base: Rcscan/Qwen3.5-2.8B-V0.1
Hiperparâmetros otimizados:
- Rank (r): 32 (aumentado para capturar melhor representação)
- Alpha: 64
- Módulos alvo: Todas as projeções lineares (atenção + MLP)
- Quantização: NF4 4-bit (BitsAndBytes)
- Scheduler: Cosine com warmup
- Upload: Envio automático para repositório PRIVADO no Hugging Face
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning para Qwen3.5-2.8B PT-BR")
    parser.add_argument("--base_model", type=str, default="Rcscan/Qwen3.5-2.8B-V0.1", help="ID do modelo base no HF")
    parser.add_argument("--dataset_path", type=str, default="./dados/dataset_ptbr_clean.jsonl", help="Caminho do dataset JSONL")
    parser.add_argument("--output_dir", type=str, default="./qwen3.5-2.8b-qlora-ptbr", help="Diretório de saída local")
    parser.add_argument("--hf_repo_id", type=str, required=False, help="ID do repositório no HF (ex: 'seu-usuario/qwen3.5-2.8b-v0.2-privado')")
    parser.add_argument("--hf_token", type=str, default=None, help="Token com permissão de escrita no HF")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de treinamento")
    parser.add_argument("--lr", type=float, default=2e-4, help="Taxa de aprendizado")
    parser.add_argument("--lora_r", type=int, default=32, help="Rank do LoRA")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha do LoRA")
    return parser.parse_args()

def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    
    print("="*60)
    print("Iniciando Fine-Tuning QLoRA do Qwen3.5-2.8B")
    print(f"Modelo Base: {args.base_model}")
    print(f"Dataset:     {args.dataset_path}")
    print(f"LoRA Rank:   {args.lora_r} | Alpha: {args.lora_alpha} | LR: {args.lr}")
    print("="*60)

    # 1. Configuração de Quantização 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Carregamento do Tokenizer e Modelo
    print("[*] Carregando modelo base em 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Configuração do Adaptador LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Carregamento e Preparação do Dataset
    print(f"[*] Carregando dataset de {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def formatar_para_treino(batch):
        textos = []
        for msgs in batch["messages"]:
            texto = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            textos.append(texto)
        return {"text": textos}

    dataset = dataset.map(formatar_para_treino, batched=True)

    # 5. Argumentos de Treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none"
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[*] Iniciando o treinamento...")
    trainer.train()

    # 7. Salvamento Local
    print(f"[*] Salvando adaptador em {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Upload Privado no Hugging Face Hub (se configurado)
    if args.hf_repo_id:
        if not token:
            print("[AVISO] hf_repo_id fornecido, mas HF_TOKEN não configurado. Pule o push_to_hub ou configure seu token.")
        else:
            print(f"[*] Enviando modelo para o repositório PRIVADO: {args.hf_repo_id}...")
            trainer.model.push_to_hub(
                args.hf_repo_id,
                private=True,
                token=token
            )
            tokenizer.push_to_hub(
                args.hf_repo_id,
                private=True,
                token=token
            )
            print(f"[SUCESSO] Modelo publicado privadamente em: https://huggingface.co/{args.hf_repo_id}")

    print("[OK] Processo finalizado!")

if __name__ == "__main__":
    main()

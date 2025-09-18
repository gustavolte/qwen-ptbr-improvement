#!/usr/bin/env python3
"""
Utilitário para upload de datasets ou modelos no Hugging Face
utilizando apenas a API REST do Hugging Face (sem dependências externas).
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

HF_API_BASE = "https://huggingface.co/api"

def criar_repositorio(repo_id: str, repo_type: str = "dataset", private: bool = True, token: Optional[str] = None):
    """
    Cria um repositório no Hugging Face (dataset ou model) como privado.
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN não encontrado nas variáveis de ambiente nem passado como argumento.")

    url = f"{HF_API_BASE}/repos/create"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Antigravity-HF-Uploader/1.0"
    }
    
    parts = repo_id.split("/")
    name = parts[-1]
    organization = parts[0] if len(parts) > 1 else None

    payload = {
        "name": name,
        "type": repo_type,
        "private": private
    }
    if organization:
        payload["organization"] = organization

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[OK] Repositório criado com sucesso: {data.get('url', repo_id)}")
            return data
    except urllib.error.HTTPError as e:
        if e.code == 409:
            print(f"[INFO] O repositório '{repo_id}' já existe. Continuando...")
            return {"url": f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"}
        else:
            erro = e.read().decode("utf-8")
            print(f"[ERRO] Falha ao criar repositório ({e.code}): {erro}")
            raise

def fazer_upload_arquivo(repo_id: str, caminho_local: str, caminho_remoto: str, repo_type: str = "dataset", token: Optional[str] = None):
    """
    Faz upload de um arquivo para o repositório no Hugging Face usando a API de upload direto.
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN não encontrado.")

    prefix = "datasets/" if repo_type == "dataset" else "models/"
    url = f"https://huggingface.co/api/{prefix}{repo_id}/upload/main/{caminho_remoto}"

    with open(caminho_local, "rb") as f:
        file_bytes = f.read()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "User-Agent": "Antigravity-HF-Uploader/1.0"
    }

    req = urllib.request.Request(url, data=file_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"[OK] Upload concluído: {caminho_remoto} -> {repo_id}")
            return True
    except urllib.error.HTTPError as e:
        erro = e.read().decode("utf-8")
        print(f"[ERRO] Falha no upload ({e.code}): {erro}")
        raise

if __name__ == "__main__":
    import sys
    print("Script utilitário para API Hugging Face carregado.")
    print("Uso via Python import ou CLI com token de autenticação.")

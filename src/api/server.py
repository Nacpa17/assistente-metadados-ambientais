"""API REST do modelo fine-tuned — agora servindo também a interface web.

Novidades desta versão:
- Serve a interface de chat (pasta ../frontend) no MESMO endereço da API,
  em "/". Assim não há problema de CORS e basta abrir http://localhost:8000.
- Ao iniciar, ABRE O NAVEGADOR automaticamente na interface.
  (desative com a variável de ambiente IC_NO_BROWSER=1)

Subir tudo de uma vez:
    python -m src.api.server
ou, se preferir o uvicorn direto:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000

Variáveis de ambiente:
    IC_CONFIG_PATH   caminho para configs/exp01.yaml (obrigatório)
    IC_PORT          porta (default 8000) — usada para abrir o navegador
    IC_NO_BROWSER    se definida, NÃO abre o navegador automaticamente
    IC_API_KEY       se setada, exige header X-API-Key igual
    IC_CORS_ORIGINS  CSV de origens (default "*")
    IC_MAX_HISTORY   máx. de turnos preservados (default 8)
    IC_RAG           1 = liga o RAG (arquitetura híbrida); 0 = desliga.
                     Sem a variável, vale `rag.enabled` do config.

Com RAG, cada resposta traz as fichas consultadas ("sources"); no streaming
elas chegam num evento `sources` antes do texto.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.api.sse import sse_event
from src.common.io import load_yaml, setup_logger
from src.common.prompts import build_chat_prompt, DEFAULT_SYSTEM_PROMPT

log = setup_logger("api.server")

# Limite do prompt sem RAG (o mesmo da fase 2).
MAX_INPUT_TOKENS = 1536

# pasta da interface: <raiz do projeto>/frontend
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
HAS_FRONTEND = FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").exists()


class AppState:
    model = None
    tokenizer = None
    rag = None                      # RagPipeline quando o RAG está ligado
    rag_error: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    gen_defaults: dict = {}
    requests_total: int = 0
    requests_errors: int = 0
    tokens_generated_total: int = 0
    last_latency_s: float = 0.0


def rag_requested(cfg: dict) -> bool:
    """IC_RAG (se definida) manda; senão vale rag.enabled do config."""
    env = os.getenv("IC_RAG")
    if env is not None and env.strip():
        return env.strip().lower() in ("1", "true", "sim", "yes", "on")
    return bool((cfg.get("rag") or {}).get("enabled", False))


def load_rag(cfg: dict) -> None:
    """Carrega o índice RAG; se faltar, o servidor sobe sem RAG (com aviso)."""
    STATE.rag, STATE.rag_error = None, None
    if not rag_requested(cfg):
        return
    try:
        from src.rag.pipeline import RagPipeline
        STATE.rag = RagPipeline.from_config(cfg)
        log.info("RAG ligado: %s (%s documentos, top_k=%d)",
                 STATE.rag.settings.index_path,
                 STATE.rag.index.info.get("n_docs", len(STATE.rag.index.docs)),
                 STATE.rag.settings.top_k)
    except Exception as e:
        STATE.rag_error = str(e)
        log.warning("RAG pedido, mas não carregou — seguindo SEM RAG. "
                    "Motivo: %s", e)


STATE = AppState()
MAX_HISTORY = int(os.getenv("IC_MAX_HISTORY", "8"))


def _open_browser_soon():
    """Abre o navegador na interface poucos segundos após o servidor subir."""
    if os.getenv("IC_NO_BROWSER") or not HAS_FRONTEND:
        return
    port = os.getenv("IC_PORT", "8000")
    url = f"http://localhost:{port}/"

    def _run():
        try:
            import webbrowser
            time.sleep(2.0)  # dá tempo do uvicorn começar a aceitar conexões
            webbrowser.open(url)
            log.info("Interface aberta no navegador: %s", url)
        except Exception as e:
            log.warning("Não consegui abrir o navegador (%s). "
                        "Acesse manualmente: %s", e, url)

    threading.Thread(target=_run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = os.getenv("IC_CONFIG_PATH")
    if not cfg_path:
        raise RuntimeError(
            "IC_CONFIG_PATH não definido — aponte para configs/exp01.yaml")
    cfg = load_yaml(cfg_path)
    mcfg = cfg["model"]
    qcfg = cfg.get("quantization", {})
    # índice RAG antes do LLaMA: se faltar, o aviso aparece em segundos
    load_rag(cfg)

    log.info("Inicializando modelo a partir de %s", cfg_path)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_compute_dtype=dtype_map[qcfg.get("bnb_4bit_compute_dtype",
                                                  "bfloat16")],
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant",
                                                True)),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
    )
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base"], quantization_config=bnb, device_map="auto",
        trust_remote_code=bool(mcfg.get("trust_remote_code", False)))
    model = PeftModel.from_pretrained(base, mcfg["output_dir"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(mcfg["output_dir"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # prompt longo demais: corta o início (histórico antigo), nunca a pergunta
    tokenizer.truncation_side = "left"

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    STATE.gen_defaults = cfg.get("generation", {})
    log.info("Modelo pronto.")
    if HAS_FRONTEND:
        _open_browser_soon()
    else:
        log.warning("Pasta frontend não encontrada em %s; a interface não "
                    "será servida.", FRONTEND_DIR)
    yield
    STATE.model = None
    STATE.tokenizer = None


app = FastAPI(title="Assistente DataMap — API", version="2.1", lifespan=lifespan)

cors_origins = os.getenv("IC_CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


async def check_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("IC_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="X-API-Key inválido")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    history: list[Message] = Field(default_factory=list)
    message: str | None = Field(default=None)
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=1.0, le=2.0)


class ChatResponse(BaseModel):
    reply: str
    latency_s: float
    tokens_generated: int
    sources: list[dict] = Field(default_factory=list)


def truncate_history(history: list[Message], max_turns: int) -> list[Message]:
    return history if len(history) <= max_turns else history[-max_turns:]


def _count_tokens(text: str) -> int:
    return len(STATE.tokenizer(text)["input_ids"])


def build_prompt(req: ChatRequest) -> tuple[str, list[dict], int]:
    """(prompt, fontes, limite de tokens do prompt) para a requisição."""
    history = truncate_history(req.history, MAX_HISTORY)
    if req.message is None:
        if not history or history[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="Forneça 'message' ou termine 'history' com role=user.")
        new_user_message = history[-1].content
        history = history[:-1]
    else:
        new_user_message = req.message
    hist = [m.model_dump() for m in history]
    if STATE.rag is not None:
        res = STATE.rag.build_prompt(STATE.system_prompt, new_user_message,
                                     history=hist, count_tokens=_count_tokens)
        return res.prompt, res.sources, STATE.rag.settings.max_input_tokens
    prompt = build_chat_prompt(STATE.system_prompt, hist,
                               new_user_message=new_user_message)
    return prompt, [], MAX_INPUT_TOKENS


def build_inputs(req: ChatRequest):
    prompt, sources, max_len = build_prompt(req)
    inputs = STATE.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_len).to(STATE.model.device)
    return inputs, sources


def gen_kwargs(req: ChatRequest) -> dict:
    return {
        "max_new_tokens": req.max_new_tokens,
        "do_sample": req.temperature > 0.0,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "pad_token_id": STATE.tokenizer.eos_token_id,
    }


# ---- rotas de API (registradas ANTES do static em "/") ----

@app.get("/healthz", tags=["health"])
def healthz():
    rag = {"enabled": STATE.rag is not None}
    if STATE.rag is not None:
        rag.update({"index": STATE.rag.settings.index_path,
                    "n_docs": STATE.rag.index.info.get(
                        "n_docs", len(STATE.rag.index.docs)),
                    "top_k": STATE.rag.settings.top_k})
    elif STATE.rag_error:
        rag["error"] = STATE.rag_error
    return {"status": "ok", "model_loaded": STATE.model is not None,
            "frontend": HAS_FRONTEND, "rag": rag}


@app.get("/metrics", tags=["health"])
def metrics():
    return {
        "requests_total": STATE.requests_total,
        "requests_errors": STATE.requests_errors,
        "tokens_generated_total": STATE.tokens_generated_total,
        "last_latency_s": round(STATE.last_latency_s, 3),
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"],
          dependencies=[Depends(check_api_key)])
def chat(req: ChatRequest):
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    STATE.requests_total += 1
    try:
        t0 = time.time()
        inputs, sources = build_inputs(req)
        with torch.inference_mode():
            out = STATE.model.generate(**inputs, **gen_kwargs(req))
        dt = time.time() - t0
        new_ids = out[0, inputs["input_ids"].shape[-1]:]
        text = STATE.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        STATE.tokens_generated_total += int(new_ids.shape[0])
        STATE.last_latency_s = dt
        return ChatResponse(reply=text, latency_s=round(dt, 3),
                            tokens_generated=int(new_ids.shape[0]),
                            sources=sources)
    except HTTPException:
        STATE.requests_errors += 1
        raise
    except Exception as e:
        STATE.requests_errors += 1
        log.exception("Erro em /chat")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["chat"],
          dependencies=[Depends(check_api_key)])
async def chat_stream(req: ChatRequest, request: Request):
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    inputs, sources = build_inputs(req)
    streamer = TextIteratorStreamer(
        STATE.tokenizer, skip_prompt=True, skip_special_tokens=True)
    kw = gen_kwargs(req)
    kw.update({"streamer": streamer, **inputs})

    def _run():
        try:
            with torch.inference_mode():
                STATE.model.generate(**kw)
        except Exception as e:
            log.exception("Erro no thread de geração: %s", e)

    threading.Thread(target=_run, daemon=True).start()
    STATE.requests_total += 1

    async def event_gen():
        # Texto em JSON: quebras de linha não partem o evento SSE
        # (antes, o pedaço seguinte a um "\n" sumia na interface).
        try:
            if sources:
                yield sse_event(sources, event="sources")
            for piece in streamer:
                if await request.is_disconnected():
                    log.info("Cliente desconectou; abortando stream")
                    break
                yield sse_event(piece)
                await asyncio.sleep(0)
            yield sse_event("[END]", event="done")
        except Exception as e:
            STATE.requests_errors += 1
            log.exception("Erro no stream")
            yield sse_event(str(e), event="error")

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---- interface web (montada por ÚLTIMO para não engolir as rotas acima) ----
if HAS_FRONTEND:
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True),
              name="frontend")
else:
    @app.get("/", tags=["health"])
    def _root():
        return {"status": "ok",
                "aviso": "interface não encontrada; rode a partir da raiz do "
                         "projeto para servir a pasta ./frontend"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("IC_PORT", "8000"))
    if not os.getenv("IC_CONFIG_PATH"):
        # padrão sensato para rodar com um comando só
        os.environ["IC_CONFIG_PATH"] = "configs/exp01.yaml"
        log.info("IC_CONFIG_PATH não definido; usando configs/exp01.yaml")
    log.info("Subindo servidor em http://localhost:%d (a interface abre "
             "sozinha)", port)
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=port, reload=False)

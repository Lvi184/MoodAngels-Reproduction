from __future__ import annotations
from typing import Dict, Any
try:
    from fastapi import FastAPI
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install API extras: pip install .[api]") from exc
from .pipeline import MoodAngelsPipeline

app = FastAPI(title="MoodAngels Research API")
pipe = MoodAngelsPipeline("data/syn_train.json")

@app.post("/diagnose")
def diagnose(payload: Dict[str, Any], agent: str="multi"):
    res = pipe.diagnose_dict(payload, agent=agent)
    import dataclasses
    return dataclasses.asdict(res)

from transformers import AutoProcessor, AutoModelForCTC, AutoModelForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, BertModel
from fastapi import FastAPI, UploadFile, File

import json
import torch
import torchaudio
import torch.nn as nn
import fastapi

from utils import ASRSAInference

app = FastAPI()

asr = ASRSAInference()

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    try:
        transcription = asr.asr_sa(file.file)
        return transcription
    except Exception as e:
        return {"error": str(e)}

@app.get("/transcribe")
def transcribe(file: UploadFile = File(...)):
    try:
        transcription = asr.asr_sa(file.file)
        return transcription
    except Exception as e:
        return {"error": str(e)}

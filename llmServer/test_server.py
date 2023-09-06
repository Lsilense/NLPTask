import os
from fastapi import FastAPI
import torch
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware

tokenizer = AutoTokenizer.from_pretrained("/data/zhangxin/LLMModel/model_test/chatglm6b-reli", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/zhangxin/LLMModel/model_test/chatglm6b-reli", trust_remote_code=True)
model = model.eval()
MAX_HISTORY = 3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    worker_id = os.getpid()
    gpu_id = worker_id % 4
    print(f"Worker {worker_id} is assigned to GPU {gpu_id}")
    model.to(torch.device("cuda:{}".format(gpu_id)))


@app.post("/chat/")
async def create_workers(arg_dict: dict):
    result = {"query": "", "response": "", "success": False}
    try:
        text = arg_dict["query"]
        ori_history = arg_dict["history"]
        print("Query - {}".format(text))
        if len(ori_history) > 0:
            print("History - {}".format(ori_history))
        history = ori_history[-MAX_HISTORY:]
        history = [tuple(h) for h in history]
        response, history = model.chat(tokenizer, text, history)
        print("Answer - {}".format(response))
        ori_history.append((text, response))
        result = {"query": text, "response": response, "history": ori_history, "success": True}
    except Exception as e:
        print(f"error: {e}")
    return result


if __name__ == "__main__":
    uvicorn.run("test_server:app", host="127.0.0.1", port=8000, workers=4)

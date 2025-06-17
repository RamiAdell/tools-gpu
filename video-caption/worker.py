# worker.py
import torch
from whisper import load_model

class GPUWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
    def load_model(self):
        if self.model is None:
            torch.cuda.init()
            self.model = load_model("small", device=self.device)
            
    def process(self, input_data):
        if self.device == "cuda":
            torch.cuda.set_device(0)
        return self.model.transcribe(input_data)
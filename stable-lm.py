from __future__ import annotations
from transformers import StoppingCriteria
import torch
import modal

# Cache the model in a shared volume to avoid downloading each time
volume = modal.SharedVolume().persist("stable-lm-model-cache-vol")
cache_path = "/vol/cache"

# Select model
model_name = "stabilityai/stablelm-tuned-alpha-7b" 

# Other Options:
# "stabilityai/stablelm-base-alpha-7b"
# "stabilityai/stablelm-tuned-alpha-7b"
# "stabilityai/stablelm-base-alpha-3b"
# "stabilityai/stablelm-tuned-alpha-3b"

# Install dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "torch",
        "transformers",
        "safetensors",
        "sentencepiece",
    )
)

# Declare Modal stub
stub = modal.Stub(name="stable-lm", image=image)

# StableLM Tuned stop-on-tokens class from readme https://github.com/Stability-AI/StableLM
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Form chat prompt string in StableLM Tuned chat format
def form_chat_prompt (prompt):
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    return f"{system_prompt}<|USER|>{prompt}<|ASSISTANT|>"


# Declare class to represent the Modal container
@stub.cls(
    gpu="A10G",
    shared_volumes={cache_path: volume},
    container_idle_timeout=500,
)
class StableLM:
    # Initialize the model and tokenizer - only called once when the container first starts
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Select "big model inference" parameters
        torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
        load_in_8bit = False #@param {type:"boolean"}
        device_map = "auto"

        # Intialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)

        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, torch_dtype),
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            offload_folder="./offload",
            cache_dir=cache_path,
        )
              
        self.model = model
        self.tokenizer = tokenizer

    # Get the LLM chatbot completion for a prompt
    @modal.method()
    def run_inference(self, prompt):
        from transformers import StoppingCriteriaList, TextStreamer

        # Form chat prompt string
        prompt = form_chat_prompt(prompt)

        # Setup streamer to print each token as it is generated
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        # Sampling args
        max_new_tokens = 1024
        temperature = 0.5
        top_k = 0
        top_p = 0.9
        do_sample = True

        # Create `generate` inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)

        # Generate
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        # Extract out only the completion tokens
        completion_tokens = tokens[0][inputs['input_ids'].size(1):]
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return completion

# Run with "modal run stable-lm.py"   
@stub.local_entrypoint()
def main():
    print('starting')
    prompt = "Write me a poem about your own existence."
    result = StableLM().run_inference.call(prompt)
    print('User: ', prompt)
    print('SableLM: ', result)

# Serve with "modal serve stable-lm.py"
# Query with GET https://{modal-app}-get_chat_completion.modal.run/?prompt="hello world"
@stub.function()
@modal.web_endpoint()
def get_chat_completion(prompt: str):
    result = StableLM().run_inference.call(prompt)
    return {"response": result}
from __future__ import annotations
import torch
import modal

# Cache the model in a shared volume to avoid downloading each time
volume = modal.SharedVolume().persist("stable-lm-model-cache-vol")
cache_path = "/vol/cache"

# Select model
# Options: "stablelm-base-alpha-7b", "stablelm-tuned-alpha-7b", "stablelm-base-alpha-3b", "stablelm-tuned-alpha-3b"
model_name = "stabilityai/stablelm-tuned-alpha-7b"

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

# Declare class to represent the Modal container
@stub.cls(
    gpu="A10G", # Could also use A100, but A10G is cheaper and plenty fast for a 7b param model
    shared_volumes={cache_path: volume}, # Mount the cached model volume
    container_idle_timeout=500, # How long to keep the model warm before shutting down the container
    concurrency_limit=1, # Due to such long startup, only allow one container to be running at a time
)
class StableLM:
    from transformers import StoppingCriteria

    # Initialize the model and tokenizer - only called once when the container first starts
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Select "big model inference" parameters
        torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
        load_in_8bit = False
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
        
        # Assign as class variables
        self.model = model
        self.tokenizer = tokenizer

    # StableLM Tuned stopping criteria class from readme https://github.com/Stability-AI/StableLM
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # Form chat prompt string in StableLM Tuned chat format
    def form_chat_prompt (self, prompt):
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """

        return f"{system_prompt}<|USER|>{prompt}<|ASSISTANT|>"

    # Get the LLM chatbot completion for a prompt
    @modal.method()
    def run_inference(self, prompt):
        from threading import Thread
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        # Form chat prompt string
        prompt = self.form_chat_prompt(prompt)

        # Setup streamer to print each token as it is generated
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        # Sampling args
        max_new_tokens = 1024
        temperature = 0.5
        top_k = 0
        top_p = 0.9
        do_sample = True

        # Create `generate` inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)

        # For streaming: run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([self.StopOnTokens()])
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            print(new_text, end="")
            yield new_text

@stub.function()
@modal.web_endpoint()
def get_chat_completion(prompt: str):
    print('Received prompt: ', prompt)

    result = ""
    for new_text in StableLM().run_inference.call(prompt):
        result += new_text
    
    return {"response": result}

@stub.function()
@modal.web_endpoint()
async def get_chat_completion_stream(prompt: str):
    from fastapi.responses import StreamingResponse
    
    print('Received prompt: ', prompt)

    def response_stream():
        for new_text in StableLM().run_inference.call(prompt):
            yield new_text

    return StreamingResponse(
        response_stream(), media_type="text/event-stream"
    )

@stub.local_entrypoint()
def main():
    prompt = "Write me a poem about your own existence."
    result = ""
    for new_text in StableLM().run_inference.call(prompt):
        result += new_text

    print('User: ', prompt)
    print('Stable: ', result)


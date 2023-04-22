from __future__ import annotations
import os
import torch
import modal
from fastapi import FastAPI

# Cache the model in a shared volume to avoid downloading each time
volume = modal.SharedVolume().persist("vicuna-model-cache-vol")
cache_path = "/vol/cache"

# Private due to licensing restrictions, 
# Follow guide here to generate your own from the base Llama weights -  https://github.com/lm-sys/FastChat/tree/main#vicuna-13b
model_name = "triestpa/generated-vicuna-13b"

# Install dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "asyncio",
        "accelerate",
        "bitsandbytes",
        "gradio",
        "fastapi",
        "ftfy",
        "torch",
        "tokenizers",
        "transformers",
        "triton",
        "safetensors",
        "sentencepiece",
    )
    .pip_install("xformers", pre=True)
)

# Declare Modal stub
stub = modal.Stub(name="vicuna", image=image)

# Declare FastAPI app
web_app = FastAPI()

# Declare class to represent the Modal container
@stub.cls(
    gpu="A100", # Vicuna 13B requires 28GB of GPU memory, so A100 is required. 8-bit 13B and full-size 7B can run on A10G.
    shared_volumes={cache_path: volume}, # Mount the cached model volume
    container_idle_timeout=500, # How long to keep the model warm before shutting down the container
    secret=modal.Secret.from_name("huggingface-secret"), # Huggingface token secret for accessing Vicuna weights
)
class Vicuna:
    from transformers import StoppingCriteria

    # Initialize the model and tokenizer - only called once when the container first starts
    def __enter__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # HF token is needed to access the private Vicuna weights
        hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

        # Select "big model inference" parameters
        torch_dtype = "float16"
        load_in_8bit = False
        device_map = "auto"

        # Intialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False, 
            cache_dir=cache_path, 
            use_auth_token=hugging_face_token)

        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True, 
            torch_dtype=getattr(torch, torch_dtype),
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            offload_folder="./offload",
            cache_dir=cache_path,
            use_auth_token=hugging_face_token
        )
        
        # Assign as class variables
        self.model = model
        self.tokenizer = tokenizer

    # Form chat prompt string in Vicuna Tuned chat format
    def form_chat_prompt (self, prompt):
        system_prompt = """A chat between a curious user and an artificial intelligence assistant.
           The assistant gives helpful, detailed, and polite answers to the user's questions.
        """

        return f"{system_prompt}</s>Human: {prompt}</s>Assistant:"

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
    for new_text in Vicuna().run_inference.call(prompt):
        result += new_text
    
    return {"response": result}

@stub.function()
@modal.web_endpoint()
async def get_chat_completion_stream(prompt: str):
    from fastapi.responses import StreamingResponse
    
    print('Received prompt: ', prompt)

    def response_stream():
        for new_text in Vicuna().run_inference.call(prompt):
            yield new_text

    return StreamingResponse(
        response_stream(), media_type="text/event-stream"
    )

@stub.local_entrypoint()
def main():
    prompt = "Write me a poem about your own existence."
    result = ""
    for new_text in Vicuna().run_inference.call(prompt):
        result += new_text

    print('User: ', prompt)
    print('Vicuna: ', result)

@stub.function()
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    with gr.Blocks() as interface:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def get_response(message, chat_history):
            result = ""
            for new_text in Vicuna().run_inference.call(message):
                result += new_text
            return result

        def respond(message, chat_history):
            bot_message = get_response(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/")

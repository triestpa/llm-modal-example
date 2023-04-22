# LLM Modal Starter

Simple [Modal](https://modal.com/) app example for serving [StableLM](https://github.com/stability-AI/stableLM/) and [Vicuna](https://github.com/lm-sys/FastChat).

![](./images/readme_header.png)
</br>*Image generated with [Stable Diffusion XL](https://clipdrop.co/stable-diffusion)*

# Usage
1. Create [Modal](https://modal.com/) Account
1. Clone App:
    ```bash
    git clone https://github.com/triestpa/llm-modal-example.git
    cd llm-modal-example
    ```
1. Setup Modal In Local Env:
    ```bash
    pip install modal-client
    modal token new
    ```
1. Serve App:
    ```bash
    modal serve stable-lm.py
    ```
    or
    ```bash
    modal serve vicuna.py
    ```
1. Generate Completion:
    ```
    Standard:
    HTTP GET https://{modal-app}-get-chat-completion-dev.modal.run?prompt="hello world"

    Streaming:
    HTTP GET https://{modal-app}-get-chat-completion-stream-dev.modal.run?prompt="hello world"
    ```
    You can also call the model directly with `modal run stable-lm.py` (or `modal run vicuna.py`), which will run the `main` function, containing a prompt you can modify.

# Startup & Latency
- The first time it runs, it will take a while for the endpoint to work, as it needs to first download the model and load it into memory.  After the first run, the model is cached in persistent storage and the download step will be skipped.
- On subsequent starts, it will take a couple minutes for the first request to respond while the model is being loaded into memory.  
- After the first request for each run, each completion will be much faster, nearly instant for short responses.
- If 500 seconds pass without a request, [the container will shut down](https://modal.com/docs/guide/cold-start) (you'll stop being billed) - and on the next request it will take longer as it will need to load the model back into memory.

# Accessing Model Weights
OpenLM should work out-of-the-box as it is completely publically available.

Vicuna is trained from [Llama](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), which has a restricted license that does not allow public distribution.

In order to use Vicuna, the recommended flow is:
1.  Download Llama weights
1.  Follow the instructions [here](https://github.com/lm-sys/FastChat/tree/main#vicuna-13b) to apply the Vicuna weight deltas to the Llama weights.
1.  Upload the resulting Vicuna weights to a HuggingFace model repository.
1. Generate a [HuggingFace access token](https://huggingface.co/docs/hub/security-tokens) and add it as a [Modal secret](https://modal.com/docs/guide/secrets) called `huggingface-secret`.
1.  Replace `model_name = "triestpa/generated-vicuna-13b"` with the path for the HuggingFace repository.
1. The `modal run vicuna.py` command should now be able to download and run inference on the Vicuna weights you generated.


# License

Open source MIT license, use this for anything you want.

```
MIT License

Copyright (c) [2023] [Patrick Triest]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

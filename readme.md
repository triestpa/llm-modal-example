# StableLM Modal Starter

![](./images/readme_header.png)

Simple [Modal](https://modal.com/) app example for serving [StableLM](https://github.com/stability-AI/stableLM/).

# Usage
1. Create Modal Account
1. Clone App:
    ```bash
    git clone https://github.com/triestpa/stable-lm-modal-example.git
    cd stable-lm-modal-example
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
1. Generate Completion:
    ```
    HTTP GET https://{modal-app}-get_chat_completion.modal.run/?prompt="hello world"
    ```

# Notes
- The first time it runs, it will take a while for the endpoint to work, as it needs to first download the model and load it into memory.  After the first run, the model is cached in persistent storage and the download step will be skipped.
- On subsequent starts, it will take a couple minutes for the first request to respond while the model is being loaded into memory.  
- After the first request for each run, each completion will be much faster, nearly instant for short responses.
- If 500 seconds pass without a request, [the container will shut down](https://modal.com/docs/guide/cold-start) (you'll stop being billed) - and on the next request it will take longer as it will need to load the model back into memory.
- On the command line and in the Modal app logs, the tokens will be printed in real-time while the response is being generated.
- You can also call the model directly with `modal run stable-lm.py`, which will run the `main` function, containing a prompt you can modify.

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
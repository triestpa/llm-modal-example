# StableLM Modal Starter

![](./images/readme_header.png)

Simple [Modal](https://modal.com/) app example for serving [StableLM](https://github.com/stability-AI/stableLM/).

# Usage

1. Create Modal Account
1. Setup Modal In Local Env:
    ```bash
    pip install modal-client
    modal token new
    ```
1. Serve app:
    ```bash
    modal serve stable-lm.py
    ```
1. Generate completion with:
    ```
    HTTP GET https://{modal-app}-get_chat_completion.modal.run/?prompt="hello world"
    ```

# Usage Notes
- The first time it runs, it will take a while for the endpoint to work, as it needs to first download the model and load it into memory.  After the first run, the model is be cached in persistent storage and the download step will be skipped.
- On subsequent starts, it will take a couple minutes for the first request to respond, as the model needs to be loaded into memory.  
- After the first request for each run, each completion will be much faster, nearly instant for short responses.
- If 500 seconds pass without a request (configurable in the code), the container will shut down (you'll stop being billed) - and on the next request it will take longer as it will need to load the model back into memory.
- On the command line and in the Modal app logs, the tokens will be printed in real-time while the response is being generated.
- You can also call the model directly with `modal run stable-lm.py`, which will run the `main` function, containing a prompt you can modify.

# License

Open source MIT license, use this for anything you want.
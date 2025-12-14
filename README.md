# DJ Cara Server

This project runs a DJ Voice generation server using Coqui XTTS for speech synthesis and a Llama-3 model for text generation.

## Prerequisites

- Python 3.9+
- `ffmpeg` installed and available in your system PATH (required for TTS audio processing).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Python dependencies:**

    **For Mac Users (Apple Silicon / M-Series):**
    To enable GPU acceleration for the LLM, you need to install `llama-cpp-python` with Metal support:
    ```bash
    CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements.txt
    ```

    **For Other Users (Linux/Windows with CUDA):**
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements.txt
    ```

    **CPU Only:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Server

Run the server using the following command:

```bash
python server.py --start_llm
```

*   `--start_llm`: Flag to start the LLM service (optional, if you only want TTS you can omit this).
*   `--port`: Specify the port (default: 8008).
*   `--wav`: Path to the reference wav file for voice cloning (default: `dj_cara.wav`).

### Automatic Model Download

When you run the server for the first time, it will check if the required models are present.
*   **XTTS v2**: Checks the default Coqui TTS model path.
*   **Llama 3**: Checks `./models/llm/llama.gguf`.

If the models are missing, the script will prompt you to download them automatically.

*   The LLM model (~5GB) will be downloaded to `./models/llm/`.
*   The XTTS model will be downloaded to your user data directory managed by Coqui TTS.

## API Endpoints

### 1. `/speak` (POST)
Generates speech from text.

**Payload:**
```json
{
  "text": "Hello, this is DJ Cara!"
}
```

**Response:** Audio file (`audio/wav`).

### 2. `/generate` (POST)
Generates text using the LLM (requires `--start_llm`).

**Payload:**
```json
{
  "text": "The next song is Firework by Katy Perry. Previous was Roar.",
  "system_prompt": "Optional system prompt override."
}
```

**Response:**
```json
{
  "text": "That was Roar! Now get ready to ignite the night with Firework!"
}
```

### 3. `/health` (GET)
Health check endpoint.

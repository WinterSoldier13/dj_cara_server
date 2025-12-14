import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from trainer.io import get_user_data_dir
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
import scipy.io.wavfile
import argparse
from llama_cpp import Llama
from logging import getLogger
import logging
import os
import sys
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

# Default paths
MODELS_DIR = "models"
LLM_DIR = os.path.join(MODELS_DIR, "llm")
TTS_DIR = os.path.join(MODELS_DIR, "tts")
LLM_PATH = os.path.join(LLM_DIR, "llama.gguf")

# XTTS Model Name
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global service instances
speech_service = None
llm_service = None

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def check_and_download_models():
    # 1. Check/Download XTTS
    # XTTS Model Manager downloads to a specific directory.
    # We can either let it be there or copy it.
    # To keep it simple, we use the path where ModelManager downloads it.

    print("Checking for XTTS model...")
    manager = ModelManager()

    # Check if model is already downloaded in the default location
    # The default location logic is a bit internal to TTS, but usually:
    default_xtts_path = os.path.join(get_user_data_dir("tts"), XTTS_MODEL_NAME.replace("/", "--"))

    if not os.path.exists(default_xtts_path):
        print(f"XTTS model not found at {default_xtts_path}")
        user_input = input("XTTS model is missing. Download it now? (y/n): ")
        if user_input.lower() == 'y':
            print("Downloading XTTS model...")
            manager.download_model(XTTS_MODEL_NAME)
            print("XTTS model downloaded.")
        else:
            print("Skipping XTTS download. Speech service might fail.")
    else:
        print(f"XTTS model found at {default_xtts_path}")

    return default_xtts_path

def check_and_download_llm():
    if not os.path.exists(LLM_DIR):
        os.makedirs(LLM_DIR)

    if not os.path.exists(LLM_PATH):
        print(f"LLM model not found at {LLM_PATH}")
        user_input = input("LLM model (Llama-3-8B-Instruct-GGUF) is missing. Download it now? (y/n): ")
        if user_input.lower() == 'y':
            print("Downloading LLM model... This might take a while.")
            try:
                # Using a specific model repo and filename
                repo_id = "MaziyarPanahi/Llama-3-8B-Instruct-GGUF"
                filename = "Llama-3-8B-Instruct.Q4_K_M.gguf"
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=LLM_DIR, local_dir_use_symlinks=False)

                # Rename to expected filename if different
                downloaded_path = os.path.join(LLM_DIR, filename)
                if downloaded_path != LLM_PATH:
                    os.rename(downloaded_path, LLM_PATH)
                print("LLM model downloaded.")
            except Exception as e:
                print(f"Failed to download LLM model: {e}")
        else:
            print("Skipping LLM download. LLM service will not work.")
    else:
        print(f"LLM model found at {LLM_PATH}")


class SpeechService:
    def __init__(self, wav_path, model_path):
        print("Loading Speech model...")
        self.wav_path = wav_path
        
        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_path, "config.json"))

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True)
        
        device = get_device()
        self.model.to(device)
        print(f"Speech model loaded on {device}.")

        print("Computing conditioning latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[self.wav_path]
        )
        print("Speech Service Ready.")

    def inference(self, text, language="en", speed=1.1, temperature=0.7):
        return self.model.inference(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            temperature=temperature,
            speed=speed
        )

class LLMService:
    def __init__(self):
        print("Loading LLM model...")

        n_gpu_layers = 0
        device = get_device()
        if device in ["cuda", "mps"]:
            n_gpu_layers = -1 # Offload all to GPU

        print(f"Initializing Llama with n_gpu_layers={n_gpu_layers} (Device: {device})")

        self.llm = Llama(
            model_path=LLM_PATH,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,       # Context window, increased slightly
            verbose=True
        )
        print("LLM Service Ready.")

    def get_llm_output(self, prompt, system_prompt=None):
        logger.info(f"Generating LLM output for prompt: {prompt}")
        
        default_system = "You are DJ Cara, a high-energy, witty, and charismatic radio host. Goal: Create a seamless, hype transition between two songs. Style: Energetic, punchy, cool. Constraints: Max 2 sentences. No emojis. Instruction: Acknowledge the vibe of the previous track briefly, then aggressively hype up the next track."
        actual_system = system_prompt if system_prompt else default_system

        # Use create_chat_completion to trigger "Instruct" mode
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": actual_system},
                {"role": "user", "content": prompt}
               ],
            max_tokens=128,
            temperature=0.7 
        )
    
        # Extract the actual message content
        response_text = output['choices'][0]['message']['content'].strip()
        logger.info(f"Generated LLM output: {response_text}")
        return response_text

@app.route('/speak', methods=['POST'])
def speak():
    if not speech_service:
        return jsonify({"error": "Speech service not initialized"}), 500

    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        out = speech_service.inference(
            text,
            speed=1.1,
            temperature=0.7
        )
        
        # Convert output tensor to wav bytes
        audio_data = out["wav"]
        # XTTS v2 is usually 24000Hz
        buffer = BytesIO()
        scipy.io.wavfile.write(buffer, 24000, audio_data)
        buffer.seek(0)
        
        return send_file(buffer, mimetype="audio/wav")
    except Exception as e:
        logger.error(f"Error in speech generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    if not llm_service:
        return jsonify({"error": "LLM service not running. Start server with --start_llm"}), 503

    data = request.json
    text = data.get('text')
    system_prompt = data.get('system_prompt') 
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response_text = llm_service.get_llm_output(text, system_prompt)
        return jsonify({"text": response_text})
    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    print("Booting up the DJ server")

    # Check and download models
    xtts_path = check_and_download_models()
    check_and_download_llm()

    parser = argparse.ArgumentParser("Run the DJ server")
    parser.add_argument('--port', type=int, default=8008, help='Port to run on')
    parser.add_argument('--wav', type=str, required=False, default="dj_cara.wav", help='Path to the speaker wav file')
    # Use action='store_true' for boolean flags
    parser.add_argument('--start_llm', action='store_true', help='Start the LLM server along with the speech service')

    args = parser.parse_args()

    # Initialize Speech Service
    try:
        speech_service = SpeechService(args.wav, xtts_path)
    except Exception as e:
        print(f"Failed to load Speech Service: {e}")
        # We might want to exit if speech is the main purpose
        # exit(1)

    # Conditionally Initialize LLM Service
    if args.start_llm:
        try:
            llm_service = LLMService()
        except Exception as e:
            print(f"Failed to load LLM Service: {e}")
            print("Continuing without LLM service.")

    print(f"DJ Server Ready on port {args.port}")
    app.run(port=args.port)

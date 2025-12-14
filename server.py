import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
import scipy.io.wavfile
import argparse
from llama_cpp import Llama
from logging import getLogger
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

LLM_PATH = "D:/models/llm/llama.gguf"
SPEECH_MODEL_PATH = "D:/models/tts/tts_models--multilingual--multi-dataset--xtts_v2"

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global service instances
speech_service = None
llm_service = None

class SpeechService:
    def __init__(self, wav_path):
        print("Loading Speech model...")
        self.wav_path = wav_path
        
        self.config = XttsConfig()
        model_path = SPEECH_MODEL_PATH
        self.config.load_json(f"{model_path}/config.json")

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True)
        
        if torch.cuda.is_available():
            self.model.cuda()
            print("Speech model loaded on GPU.")
        else:
            print("Speech model loaded on CPU.")

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
        # Note: You might want to make the model path configurable via arguments too
        self.llm = Llama(
            model_path=LLM_PATH,
            n_gpu_layers=28, # -1 means offload ALL layers to GPU
            n_ctx=1024       # Context window
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
            max_tokens=64,
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
    parser = argparse.ArgumentParser("Run the DJ server")
    parser.add_argument('--port', type=int, default=8008, help='Port to run on')
    parser.add_argument('--wav', type=str, required=False, default="dj_cara.wav", help='Path to the speaker wav file')
    # Use action='store_true' for boolean flags
    parser.add_argument('--start_llm', action='store_true', help='Start the LLM server along with the speech service')

    args = parser.parse_args()

    # Initialize Speech Service
    try:
        speech_service = SpeechService(args.wav)
    except Exception as e:
        print(f"Failed to load Speech Service: {e}")
        exit(1)

    # Conditionally Initialize LLM Service
    if args.start_llm:
        try:
            llm_service = LLMService()
        except Exception as e:
            print(f"Failed to load LLM Service: {e}")
            # We might want to continue even if LLM fails? 
            # For now, let's print and continue since speech is primary.
            print("Continuing without LLM service.")

    print(f"DJ Server Ready on port {args.port}")
    app.run(port=args.port)

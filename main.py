from transformers import AutoProcessor, MusicgenForConditionalGeneration
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import scipy.io.wavfile
import tempfile
import nest_asyncio
import uvicorn

# Load the processor and model once
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Print confirmation
print("Model and Processor Loaded")

app = FastAPI()

class MusicInput(BaseModel):
    prompts: list

@app.post("/generate-music/")
async def generate_music(input_data: MusicInput):
    try:
        # Process inputs and generate audio
        inputs = processor(
            text=input_data.prompts,
            padding=True,
            return_tensors="pt",
        )

        audio_values = model.generate(**inputs, max_new_tokens=256)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sampling_rate = model.config.audio_encoder.sampling_rate
            # Using the exact method you provided
            scipy.io.wavfile.write(
                temp_audio.name,
                rate=sampling_rate,
                data=audio_values[0, 0].numpy()
            )

            return FileResponse(
                temp_audio.name,
                media_type="audio/wav",
                filename="generated_music.wav"
            )

    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        nest_asyncio.apply()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("Server is shutting down...")
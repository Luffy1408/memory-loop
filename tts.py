# Text-to-Speech module using Edge-TTS for human-like voices
# Edge-TTS provides neural voices that sound natural and human-like

import asyncio
import edge_tts
import tempfile

# Voice options for human-like speech
TTS_VOICES = {
    'en': {
        'male': 'en-US-GuyNeural',        # Natural American English male voice
        'female': 'en-US-JennyNeural',    # Natural American English female voice
        'uk_male': 'en-GB-RyanNeural',    # British English male
        'uk_female': 'en-GB-SoniaNeural'  # British English female
    },
    'hi': {
        'male': 'hi-IN-MadhurNeural',
        'female': 'hi-IN-SwaraNeural'
    }
}


async def generate_speech_edge(text, output_path, voice='en-US-JennyNeural'):
    """Generate speech using Edge-TTS (human-like voice)."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"Edge-TTS error: {e}")
        return False


def text_to_speech(text, output_path="output.mp3", language='en', voice_type='female'):
    """
    Convert text to speech using Edge-TTS with human-like neural voices.

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        language: 'en' for English, 'hi' for Hindi
        voice_type: 'male', 'female', 'uk_male', 'uk_female'

    Returns:
        Path to the generated audio file if successful, None otherwise
    """
    try:
        # Get appropriate voice
        lang_code = 'hi' if language.lower() in ['hi', 'hindi'] else 'en'
        voice_key = voice_type if voice_type in TTS_VOICES.get(lang_code, {}) else 'female'
        voice = TTS_VOICES[lang_code][voice_key]

        # Run async TTS
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(generate_speech_edge(text, output_path, voice))
        loop.close()

        if success:
            return output_path
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


if __name__ == "__main__":
    # Test the TTS with a sample text
    text_en = "This is a microwave, you can use it to heat your food. Would you like me to give you a step by step instruction on how to use it?"

    output = text_to_speech(text_en, "micro_digimemoir.mp3", language='en', voice_type='female')

    if output:
        print(f"Audio saved to: {output}")
        print("Playing audio file...")
        # Optionally play the audio
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(output)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
    else:
        print("Failed to generate audio")
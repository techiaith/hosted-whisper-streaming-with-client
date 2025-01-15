import datetime
import numpy as np
import json
from faster_whisper import WhisperModel
import logging
import os
import wave
import webrtcvad
from scipy import signal

logger = logging.getLogger(__name__)


class AudioTranscriberServicer:
    def __init__(self):

        self.model_id = os.environ.get("MODEL", "medium")
        self.compute_type = os.environ.get("COMPUTE_TYPE", "float16")
        self.device = os.environ.get("DEVICE", "cuda")

        self.model = WhisperModel(
            self.model_id, device=self.device, compute_type=self.compute_type
        )

        logger.info("Loaded Whisper model.")
        
        # Warmup the model
        logger.info("Warming up the model...")
        _ = self.model.transcribe(np.zeros(16000, dtype=np.int16), beam_size=5)
        logger.info("Model warmup completed.")

        self.audio_frames = []
        self.sample_rate = 16000
        self.frame_duration = 30  # Frame duration in milliseconds
        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        
        self.vad_mode = int(os.environ.get("VAD_MODE", "3"))
        self.vad = webrtcvad.Vad(mode=self.vad_mode)
        self.vad_window_size = 30  # Number of frames to consider for VAD
        self.vad_frames = []
       
        self.recordings_dir = "/app/recordings"
        self.wav_file = None
        self.min_segment_duration = 3  # Minimum duration in seconds
        self.silence_threshold = 0.01

    def normalize_audio(self, audio):
        return (audio / np.max(np.abs(audio))).astype(np.float32)

    def high_pass_filter(self, audio, cutoff=100, fs=16000):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(5, normal_cutoff, btype="high", analog=False)
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def trim_silence(self, audio, threshold=0.01):
        def _trim(a):
            return a[~np.all(np.abs(a) < threshold, axis=1)]

        y = _trim(audio.reshape((len(audio), 1)))
        return y.flatten().astype(np.float32)

    def prevent_clipping(self, audio, threshold=0.95):
        max_val = np.max(np.abs(audio))
        if max_val > threshold:
            audio = audio * (threshold / max_val)
        return audio.astype(np.float32)

    async def transcribe_audio(self, websocket, audio_data):
        # Convert the audio data to a NumPy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Append the audio array to the accumulated audio frames
        self.audio_frames.append(audio_array)

        # Open the WAV file for writing if it hasn't been opened yet
        if self.wav_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_recording_{timestamp}.wav"
            filepath = os.path.join(self.recordings_dir, filename)
            self.wav_file = wave.open(filepath, "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2)
            self.wav_file.setframerate(self.sample_rate)

        # Write the audio array to the WAV file
        self.wav_file.writeframes(audio_array.tobytes())

        # Split the audio array into frames for VAD
        num_frames = len(audio_array) // self.frame_length
        for frame_idx in range(num_frames):
            frame_start = frame_idx * self.frame_length
            frame_end = frame_start + self.frame_length
            frame = audio_array[frame_start:frame_end]

            # Apply VAD to the frame
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

            if is_speech or self.vad_frames:
                self.vad_frames.append(frame)

            if not is_speech and self.vad_frames:
                # We've detected a pause, check if we should transcribe
                logger.debug("Detected a pause, checking if we should transcribe...")

                voice_segment = np.concatenate(self.vad_frames)

                segment_duration = len(voice_segment) / self.sample_rate
                is_active = np.mean(np.abs(voice_segment)) > self.silence_threshold
                logger.debug(f"Audio segment duration: {segment_duration:.2f}s")
                logger.debug(f"Mean amplitude: {np.mean(np.abs(voice_segment)):.2f}")
                logger.debug(f"Is active: {is_active}")
                if segment_duration >= self.min_segment_duration and is_active:
                    logger.info("Transcribing audio segment...")
                    voice_segment = self.normalize_audio(voice_segment)
                    voice_segment = self.high_pass_filter(voice_segment)
                    voice_segment = self.trim_silence(voice_segment)
                    voice_segment = self.prevent_clipping(voice_segment)

                    voice_segment = voice_segment.astype(np.float32)

                    # Transcribe the segment
                    segments, info = self.model.transcribe(
                        voice_segment,
                        beam_size=10,  # Increased from 5
                        temperature=0.0,  # More deterministic output
                        no_speech_threshold=0.5,  # Slightly lower than default to be more sensitive
                        word_timestamps=True,  # Enable word timestamps
                        # vad_filter=True,  # Enable VAD filter
                        language_detection_threshold=0.5,  # Enable language detection)
                    )
                    logger.debug(
                        f"Audio segment stats: duration={segment_duration:.2f}s, "
                        f"max_amplitude={np.max(np.abs(voice_segment)):.2f}, "
                        f"mean_amplitude={np.mean(np.abs(voice_segment)):.2f}"
                    )

                    for segment in segments:
                        logger.info(f"Transcription: {segment}")
                        words = []
                        if hasattr(segment, "words"):
                            words = [
                                {
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end,
                                    "probability": word.probability,
                                }
                                for word in segment.words
                            ]

                        await websocket.send(
                            json.dumps(
                                {
                                    "transcription": segment.text,
                                    "language": info.language,
                                    "start_time": segment.start,
                                    "end_time": segment.end,
                                    "words": words,
                                    "probability": segment.avg_logprob,  # Add overall segment probabi
                                }
                            )
                        )

                    # Reset the VAD frames after transcription
                    self.vad_frames = []
                elif segment_duration < self.min_segment_duration:
                    # Keep adding frames if minimum duration is not met
                    continue
                else:
                    # Discard if silence threshold is not met
                    self.vad_frames = []

        # logger.info("Transcription request completed.")

    async def on_websocket_disconnect(self, websocket, close_code, close_reason):
        # Close the WAV file when the WebSocket connection is closed
        if self.wav_file is not None:
            self.wav_file.close()
            self.wav_file = None
            logger.info("WAV file closed.")

# MPV Parakeet Transcriber

Automatically generate subtitles for media playing in MPV using NVIDIA's Parakeet ASR model, NeMo, FFmpeg, and Python. This system extracts audio from the current video, processes it, transcribes it using a powerful ASR model, and generates an SRT subtitle file.

## Features

* **On-the-fly Transcription:** Generate subtitles for currently playing media in MPV.
* **NVIDIA Parakeet ASR:** Utilizes the high-quality `nvidia/parakeet-tdt-0.6b-v2` model for accurate English transcription.
   * Automatic punctuation and capitalization.
   * Accurate word-level timestamps.
   * Efficiently transcribes long audio segments (updated to support upto 3 hours) (For even longer audios, see the speech_to_text_buffered_infer_rnnt.py script).
   * Robust performance on spoken numbers, and song lyrics transcription.
* **Audio Offset Correction:** Automatically detects and compensates for audio stream start time offsets in video files.
* **Multiple Processing Modes:**
    * **Standard Mode (`Alt+1`):** Default transcription with optimized precision (bfloat16/float16 on GPU).
    * **Python Float32 Mode (`Alt+2`):** Transcription using full float32 precision in Python for potentially higher accuracy on very subtle audio, at the cost of performance and VRAM.
    * **FFmpeg Pre-processing Mode (`Alt+3`):** Applies FFmpeg audio filters (e.g., for normalization/denoising) to the extracted audio before transcription with default Python precision.
    * **FFmpeg Pre-processing + Python Float32 Mode (`Alt+4`):** Combines FFmpeg audio filtering with float32 precision in Python.
* **Customizable:** Configure paths, keybindings, and FFmpeg filters.
* **Temporary File Management:** Handles temporary audio files.

## Prerequisites

1.  **MPV Media Player:** The script is an MPV Lua script.
2.  **Python Environment:**
    * Python 3.8+ (Python 3.12 was used in development, as per `python_exe` path).
    * A dedicated virtual environment is highly recommended.
    * **Required Python Packages:**
        * `nemo_toolkit[asr]` (which includes PyTorch with CUDA support if available, `torch`, `torchaudio`)
        * `soundfile`
        * `librosa`
        * (Potentially others depending on your NeMo installation - refer to NeMo documentation)
3.  **NVIDIA GPU (Recommended for Performance):**
    * The Parakeet model is computationally intensive. A CUDA-enabled NVIDIA GPU is strongly recommended for reasonable transcription speeds.
    * Ensure you have the appropriate NVIDIA drivers and CUDA toolkit version compatible with your PyTorch and NeMo installation.
4.  **FFmpeg & FFprobe:**
    * These command-line tools must be installed and accessible in your system's PATH, or you need to provide the full path in the Lua script configuration. They are used for audio extraction, pre-processing, and stream analysis.

## Setup

1.  **Clone the Repository (or download the files):**
    ```bash
    git clone https://github.com/Kunal926/mpv-parakeet-transcriber.git
    cd mpv-parakeet-transcriber
    ```

2.  **Python Environment Setup:**
    * Create a Python virtual environment:
        ```bash
        python -m venv .venv 
        ```
    * Activate the virtual environment:
        * Windows: `.venv\Scripts\activate`
        * Linux/macOS: `source .venv/bin/activate`
    * Install required packages:
        ```bash
        pip install -r requirements.txt
        ```
        *Note: Installing NeMo and PyTorch with the correct CUDA version can be complex. Refer to the [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) for detailed instructions.*

3.  **Configure the Lua Script (`parakeet_caption.lua`):**
    * Open `parakeet_caption.lua`.
    * Update the configuration section at the top:
        ```lua
        -- ########## Configuration ##########
        local python_exe = "C:/path/to/your/.venv/Scripts/python.exe" -- Or /path/to/your/.venv/bin/python
        local parakeet_script_path = "C:/path/to/your/parakeet_transcribe.py"
        local ffmpeg_path = "ffmpeg" -- Or "C:/path/to/ffmpeg.exe"
        local ffprobe_path = "ffprobe" -- Or "C:/path/to/ffprobe.exe"
        local temp_dir = "C:/temp" -- Ensure this directory exists and is writable

        -- Keybindings (Alt combinations are generally safer from conflicts)
        local key_binding_default = "alt+1"
        local key_binding_py_float32 = "alt+2"        
        local key_binding_ffmpeg_preprocess = "alt+3"  
        local key_binding_ffmpeg_py_float32 = "alt+4"

        local auto_load_and_cleanup_delay_seconds = 30

        -- FFmpeg audio filter chain for pre-processing modes (alt+3, alt+4)
        -- Start with simpler filters and test.
        -- Example: Gentle compression (worked in tests)
        local ffmpeg_audio_filters = "acompressor=threshold=0.1:ratio=2:attack=20:release=250"
        -- Example: Loudness normalization (single-pass, faster)
        -- local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5:linear=true" 
        -- Example: Denoising (can be slow, parameters need tuning)
        -- local ffmpeg_audio_filters = "anlmdn=s=5:p=0.002:r=0.002:m=15" 
        -- Example: Combination (test individual filters first)
        -- local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5:linear=true,anlmdn=s=5:p=0.002:r=0.002:m=15"
        -- ###################################
        ```
    * Ensure the paths to `python_exe`, `parakeet_script_path`, `ffmpeg_path`, `ffprobe_path`, and `temp_dir` are correct for your system.

4.  **Place Scripts in MPV Directory:**
    * Copy `parakeet_caption.lua` into your MPV `scripts` directory.
        * Windows: Typically `C:\Users\YourUser\AppData\Roaming\mpv\scripts\` or `C:\Users\YourUser\AppData\Roaming\mpv.net\scripts\` (for mpv.net).
        * Linux: Typically `~/.config/mpv/scripts/`.
        * macOS: Typically `~/.config/mpv/scripts/`.
    * Place `parakeet_transcribe.py` (currently `v17f - Robust Error SRT` or later from the immersive `parakeet_transcribe_v17`) in the location you specified in `parakeet_script_path`.

## Usage

1.  Play a video or audio file in MPV.
2.  Press one of the configured keybindings to start transcription:
    * **`Alt+1` (Default):**
        * Extracts the English audio track (or fallback).
        * Transcribes using the Parakeet model with default (optimized) precision on GPU.
        * Applies audio stream start offset.
    * **`Alt+2` (Python Float32):**
        * Same as default, but tells the Python script to use `--force_float32` for potentially higher ASR accuracy (slower, more VRAM).
    * **`Alt+3` (FFmpeg Pre-processing):**
        * Extracts audio.
        * Applies the FFmpeg filters defined in `ffmpeg_audio_filters` to the extracted audio.
        * Transcribes the *filtered* audio using default Python precision.
        * Applies audio stream start offset.
    * **`Alt+4` (FFmpeg Pre-processing + Python Float32):**
        * Extracts audio.
        * Applies FFmpeg filters.
        * Transcribes the *filtered* audio using `--force_float32` in Python.
3.  An OSD message will indicate that transcription has started. This process can take a significant amount of time, especially for long files or when using FFmpeg filters / float32 precision.
4.  After the `auto_load_and_cleanup_delay_seconds` (default 30 seconds, configurable in Lua), the script will attempt to:
    * Load the generated SRT file (e.g., `your_video_name.srt`) into MPV.
    * Clean up temporary audio files from the `temp_dir`.
5.  Check the MPV console (usually opened with `` ` `` (backtick)) for detailed log messages from the Lua script and the Python script.

## Python Script (`parakeet_transcribe.py`)

This script is the backend that performs the actual ASR using NeMo.
It accepts the following command-line arguments:
1.  `audio_file_path`: (Positional) Path to the temporary WAV audio file to transcribe.
2.  `srt_output_file_path`: (Positional) Path where the generated SRT file should be saved.
3.  `--audio_start_offset SECONDS`: (Optional) A float value in seconds to add to all generated timestamps. Used to correct sync issues if the extracted audio doesn't start at time 0 relative to the original video. (Default: 0.0)
4.  `--force_float32`: (Optional Flag) If present, forces the NeMo model to run in `float32` precision on the GPU, even if lower (faster) precisions like `bfloat16` or `float16` are available. This may slightly improve accuracy in some cases but is slower and uses significantly more VRAM.

## Troubleshooting

* **"SRT file not found" or "SRT file empty":**
    * Check the MPV console for errors from the Lua script or the Python script.
    * **CUDA OutOfMemoryError:** If transcribing long files (especially with `--force_float32` or complex FFmpeg filters), you might run out of GPU VRAM. The Python script should attempt to write an error SRT in this case. Try a shorter file or use a less memory-intensive mode (e.g., default `Ctrl+W`).
    * **FFmpeg Filtering Failed:** If using a pre-processing mode (`Alt+8`, `Alt+9`), the FFmpeg filtering step might fail (e.g., if filters are too complex, take too long, or there's an issue with FFmpeg itself). The log should indicate this. Try simplifying `ffmpeg_audio_filters` in the Lua script.
    * **Path Issues:** Double-check all configured paths in `parakeet_caption.lua`.
    * **Python Environment:** Ensure your Python environment is activated and has all necessary packages correctly installed (especially NeMo and a compatible PyTorch with CUDA).
    * **Permissions:** Ensure the script has permission to write to the `temp_dir` and the directory where the media file is located (for saving the SRT).
* **Timestamps are off:**
    * The script now automatically detects and applies an offset based on the selected audio stream's `start_time` in the original media. If sync is still off, ensure `ffprobe` is working correctly and providing accurate `start_time` information for your files.
* **Transcription Accuracy Issues (Missed Lines, Incorrect Words):**
    * **Audio Quality:** The cleaner the audio, the better the transcription.
    * **FFmpeg Filters:** Experiment with the `ffmpeg_audio_filters` in the Lua script. `loudnorm` can help with quiet audio. Denoisers (`anlmdn`, `afftdn`) can help with background noise but need careful tuning to avoid degrading speech. Test on short clips first.
    * **Python Precision:** Try the `--force_float32` modes (`Alt+7`, `Alt+9`) on shorter, problematic segments to see if it improves accuracy. Be mindful of VRAM usage.
    * **Model Limitations:** ASR is not perfect. Some complex audio scenarios (heavy overlap, very thick accents, very domain-specific jargon) might always be challenging for the current model.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.


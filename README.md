![Model Arch](https://img.shields.io/badge/Model%20Arch-FastConformer--TDT-blue)
![Params](https://img.shields.io/badge/Params-0.6B-brightgreen)
![Language](https://img.shields.io/badge/Language-en-orange)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)
![Library](https://img.shields.io/badge/Library-NeMo-success)
![Pipeline](https://img.shields.io/badge/Pipeline-ASR-informational)

# MPV Parakeet Transcriber

Automatically generate subtitles for media playing in MPV using NVIDIA's Parakeet ASR model, NeMo, FFmpeg, and Python. This system extracts audio from the current video, processes it, transcribes it using a powerful ASR model, and generates an SRT subtitle file.

## Features

* **On-the-fly Transcription:** Generate subtitles for currently playing media in MPV.
* **NVIDIA Parakeet ASR:** Utilizes the high-quality `nvidia/parakeet-tdt-0.6b-v2` model for accurate English transcription.
    * Automatic punctuation and capitalization.
    * Accurate word-level timestamps.
    * Efficiently transcribes long audio segments (updated to support up to 3 hours) (For even longer audios, see the `speech_to_text_buffered_infer_rnnt.py` script often found with NeMo examples).
    * Robust performance on spoken numbers, and song lyrics transcription.
* **Audio Offset Correction:** Automatically detects and compensates for audio stream start time offsets in video files.
* **Multiple Processing Modes:**
    * **Standard Mode (`Alt+4`):** Default transcription with optimized precision (bfloat16/float16 on GPU).
    * **Python Float32 Mode (`Alt+5`):** Transcription using full float32 precision in Python for potentially higher accuracy on very subtle audio, at the cost of performance and VRAM.
    * **FFmpeg Pre-processing Mode (`Alt+6`):** Applies FFmpeg audio filters (e.g., for normalization/denoising) to the extracted audio before transcription with default Python precision.
    * **FFmpeg Pre-processing + Python Float32 Mode (`Alt+7`):** Combines FFmpeg audio filtering with float32 precision in Python.
* **Customizable:** Configure paths, keybindings, and FFmpeg filters.
* **Immediate SRT Loading:** Subtitles are loaded as soon as transcription is complete.
* **Temporary File Management:** Handles temporary audio files, cleaning them up when MPV is closed.

<details>
<summary><h2>Prerequisites</h2></summary>

1.  **MPV Media Player:** The script is an MPV Lua script.
2.  **Python Environment:**
    * Python 3.8+ (Python 3.12 was used in development, as per `python_exe` path in the Lua script. However, see Troubleshooting for version considerations).
    * A dedicated virtual environment is highly recommended.
    * **Required Python Packages:**
        * `nemo_toolkit[asr]` (which includes PyTorch with CUDA support if available, `torch`, `torchaudio`)
        * `soundfile`
        * `librosa`
        * (Potentially others depending on your NeMo installation - refer to NeMo documentation, and see Troubleshooting for specific dependency notes like `sentencepiece` and `texterrors`).
3.  **NVIDIA GPU (Recommended for Performance):**
    * The Parakeet model is computationally intensive. A CUDA-enabled NVIDIA GPU is strongly recommended for reasonable transcription speeds.
    * Ensure you have the appropriate NVIDIA drivers and CUDA toolkit version compatible with your PyTorch and NeMo installation.
4.  **FFmpeg & FFprobe:**
    * These command-line tools must be installed and accessible in your system's PATH, or you need to provide the full path in the Lua script configuration. They are used for audio extraction, pre-processing, and stream analysis.
5.  **Build Tools (Potentially for specific dependencies):**
    * **CMake:** May be needed for building packages like `sentencepiece`. See Troubleshooting for version recommendations.
    * **C++ Compiler:** May be needed for building packages like `texterrors` from source (e.g., Visual Studio Build Tools on Windows).
</details>

## Setup

1.  **Clone the Repository (or download the files):**
    If you have a Git repository for this:
    ```bash
    git clone [https://github.com/Kunal926/mpv-parakeet-transcriber.git](https://github.com/Kunal926/mpv-parakeet-transcriber.git)
    cd mpv-parakeet-transcriber
    ```
    Otherwise, ensure you have `parakeet_caption.lua` and `parakeet_transcribe.py`.

2.  **Python Environment Setup:**
    * Create a Python virtual environment:
        ```bash
        python -m venv .venv
        ```
    * Activate the virtual environment:
        * Windows: `.venv\Scripts\activate`
        * Linux/macOS: `source .venv/bin/activate`
    * Install required packages (assuming you have a `requirements.txt`):
        ```bash
        pip install -r requirements.txt
        ```
        If not, install manually:
        ```bash
        pip install nemo_toolkit[asr] soundfile librosa
        ```
        *Note: Installing NeMo and PyTorch with the correct CUDA version can be complex. Refer to the [NVIDIA NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) for detailed instructions. Also, see the **Troubleshooting** section below for notes on specific Python versions and dependencies like `sentencepiece` and `texterrors`.*

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
        local key_binding_default = "Alt+4"
        local key_binding_py_float32 = "Alt+5"
        local key_binding_ffmpeg_preprocess = "Alt+6"
        local key_binding_ffmpeg_py_float32 = "Alt+7"

        -- FFmpeg audio filter chain for pre-processing modes (Alt+6, Alt+7)
        -- Start with simpler filters and test.
        -- Example: Loudness normalization (currently in your script)
        local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"
        -- Example: Gentle compression
        -- local ffmpeg_audio_filters = "acompressor=threshold=0.1:ratio=2:attack=20:release=250"
        -- Example: Denoising (can be slow, parameters need tuning)
        -- local ffmpeg_audio_filters = "anlmdn=s=5:p=0.002:r=0.002:m=15"
        -- Example: Combination (test individual filters first)
        -- local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5,anlmdn=s=5:p=0.002:r=0.002:m=15"
        -- ###################################
        ```
    * Ensure the paths to `python_exe`, `parakeet_script_path`, `ffmpeg_path`, `ffprobe_path`, and `temp_dir` are correct for your system.

4.  **Place Scripts in MPV Directory:**
    * Copy `parakeet_caption.lua` into your MPV `scripts` directory.
        * Windows: Typically `C:\Users\YourUser\AppData\Roaming\mpv\scripts\` or `C:\Users\YourUser\AppData\Roaming\mpv.net\scripts\` (for mpv.net).
        * Linux: Typically `~/.config/mpv/scripts/`.
        * macOS: Typically `~/.config/mpv/scripts/`.
    * Place `parakeet_transcribe.py` in the location you specified in `parakeet_script_path`.

## Usage

1.  Play a video or audio file in MPV.
2.  Press one of the configured keybindings to start transcription:
    * **`Alt+4` (Standard):**
        * Extracts the English audio track (or fallback).
        * Transcribes using the Parakeet model with default (optimized) precision on GPU.
        * Applies audio stream start offset.
    * **`Alt+5` (Python Float32):**
        * Same as standard, but tells the Python script to use `--force_float32` for potentially higher ASR accuracy (slower, more VRAM).
    * **`Alt+6` (FFmpeg Pre-processing):**
        * Extracts audio.
        * Applies the FFmpeg filters defined in `ffmpeg_audio_filters` to the extracted audio.
        * Transcribes the *filtered* audio using default Python precision.
        * Applies audio stream start offset.
    * **`Alt+7` (FFmpeg Pre-processing + Python Float32):**
        * Extracts audio.
        * Applies FFmpeg filters.
        * Transcribes the *filtered* audio using `--force_float32` in Python.
3.  An OSD message will indicate that transcription has started. This process can take a significant amount of time, especially for long files or when using FFmpeg filters / float32 precision.
4.  Once transcription is complete:
    * The script will immediately attempt to load the generated SRT file (e.g., `your_video_name.srt`) into MPV.
    * Temporary audio files will be cleaned up from the `temp_dir` when you close MPV.
5.  Check the MPV console (usually opened with `` ` `` (backtick)) for detailed log messages from the Lua script and the Python script.

## Vocal Isolation (Alt+8)

Press **Alt+8** to run a pre-ASR vocal separation step.

### How it works

1. FFmpeg extracts a stereo 44.1 kHz track from the current media.
2. A RoFormer model (configured via YAML + checkpoint) isolates the vocals.
3. The vocals are resampled to 16 kHz mono and fed to Parakeet for transcription.
4. The resulting subtitles are written as an SRT file and loaded into MPV.

Place model files under `weights/roformer/` following `weights/roformer/presets.yaml`.
Download the YAML + CKPT from the pcunwa Hugging Face repositories, for example:
* [voc_fv4](https://huggingface.co/pcunwa/voc-fv4)
* [mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956](https://huggingface.co/pcunwa/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956)

Switch models by editing `roformer_preset` in `parakeet_caption.lua` or by passing
`--preset` to `python separation/bsr_separate.py`.

Notes:

* **voc_fv4:** fast separation with lighter resource use.
* **mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956:** slower but higher quality isolation.

Troubleshooting:

* CUDA OOM → increase chunk size, reduce overlap, or disable `--fp16`.

## Python Transcription Script (`parakeet_transcribe.py`)

This script is the backend that performs the actual ASR using NeMo.
It accepts the following command-line arguments:
1.  `audio_file_path`: (Positional) Path to the temporary WAV audio file to transcribe.
2.  `srt_output_file_path`: (Positional) Path where the generated SRT file should be saved.
3.  `--audio_start_offset SECONDS`: (Optional) A float value in seconds to add to all generated timestamps. Used to correct sync issues if the extracted audio doesn't start at time 0 relative to the original video. (Default: 0.0)
4.  `--force_float32`: (Optional Flag) If present, forces the NeMo model to run in `float32` precision on the GPU, even if lower (faster) precisions like `bfloat16` or `float16` are available. This may slightly improve accuracy in some cases but is slower and uses significantly more VRAM.

## Troubleshooting

<details>
<summary><strong>"SRT file not found" or "SRT file empty"</strong></summary>

* First, **check the MPV console** (usually opened with `` ` ``) for any error messages from the Lua script or the Python script.
    * **Look for specific errors in the console output, such as:**
        * **CUDA OutOfMemoryError:** If transcribing long files (especially with `--force_float32` - `Alt+5` or `Alt+7`), you might run out of GPU VRAM. The Python script should attempt to write an error SRT in this case. Try a shorter file or use a less memory-intensive mode (e.g., standard `Alt+4`).
        * **FFmpeg Filtering Failed:** If using a pre-processing mode (`Alt+6`, `Alt+7`), the FFmpeg filtering step might fail (e.g., if filters are too complex, take too long, or there's an issue with FFmpeg itself). The log should indicate this. Try simplifying `ffmpeg_audio_filters` in the Lua script.
    * **If the console logs are unhelpful, or if issues persist after addressing specific errors found in logs, also verify these common configuration-related issues:**
        * **Path Issues:** Double-check all configured paths in `parakeet_caption.lua`. Ensure there are no typos and that the paths are absolute and correct for your operating system.
        * **Permissions:** Ensure the script has write permission to the `temp_dir` you configured in `parakeet_caption.lua`. Also, ensure it has permission to write the `.srt` file in the same directory as the media file.
</details>

<details>
<summary><strong>Python Environment & Dependency Issues</strong></summary>

* Ensure your Python environment is activated and has all necessary packages correctly installed (especially NeMo and a compatible PyTorch with CUDA).
* **Python Version Considerations:**
    * While the script was developed with Python 3.12, using Python versions **greater than 3.12 is currently not recommended** since official pre-compiled ONNX Runtime wheels are not immediately available for the very latest Python releases. Always check the [ONNX Runtime documentation](https://onnxruntime.ai/docs/install/) for supported Python versions.
    * Some users have reported **Python 3.11** to offer good performance and stability with NeMo and its dependencies. This could be a good version to try if you encounter issues with other versions or are looking for potentially better speed.
    * Ultimately, ensure your chosen Python version is compatible with all critical dependencies, especially `torch`, `torchaudio`, `nemo_toolkit`, and their underlying requirements like CUDA and ONNX.
* **`sentencepiece` Installation:**
    * `sentencepiece` is a common dependency for NeMo. If you encounter errors during its installation (e.g., build failures), it might be due to an issue with newer CMake versions.
    * If installing `sentencepiece` fails, try using **CMake version < 4.0 (e.g., 3.22.x, 3.25.x, or even older like 3.17.x up to 3.31.7)**. This can sometimes resolve build issues with `sentencepiece`.
* **`texterrors` Version:**
    * NVIDIA NeMo toolkit does not support texterrors version `>1.0`.
    * If you see errors related to `texterrors` or its API, you may need to install a version <1.0 or build `texterrors==0.5.1` from source:
        1.  Uninstall any existing `texterrors`: `pip uninstall texterrors`
        2.  Clone the repository: `git clone https://github.com/RuABraun/texterrors.git`
        3.  `cd texterrors`
        4.  Checkout the specific tag: `git checkout v0.5.1`
        5.  Install build tools if needed (e.g., `pip install build`).
        6.  Build and install (If it's still failing, use "x64 Native Tools Command Prompt for VS" on Windows): `pip install .`
* Refer to the NeMo documentation for the most up-to-date compatibility information.
</details>

<details>
<summary><strong>Timestamps are off</strong></summary>

* The script automatically detects and applies an offset based on the selected audio stream's `start_time` in the original media. If sync is still off, ensure `ffprobe` is working correctly and providing accurate `start_time` information for your files.
</details>

<details>
<summary><strong>Transcription Accuracy Issues (Missed Lines, Incorrect Words)</strong></summary>

* **Audio Quality:** The cleaner the audio, the better the transcription.
* **FFmpeg Filters:** Experiment with the `ffmpeg_audio_filters` in the Lua script. `loudnorm` can help with quiet audio. Denoisers (`anlmdn`, `afftdn`) can help with background noise but need careful tuning to avoid degrading speech. Test on short clips first.
* **Python Precision:** Try the `--force_float32` modes (`Alt+5`, `Alt+7`) on shorter, problematic segments to see if it improves accuracy. Be mindful of VRAM usage.
* **Model Limitations:** ASR is not perfect. Some complex audio scenarios (heavy overlap, very thick accents, very domain-specific jargon) might always be challenging for the current model.
</details>

## License

The use of the NVIDIA Parakeet model is governed by the CC-BY-4.0 license.
This project (the Lua and Python scripts) is licensed under the CC-BY-4.0 License. You can find more details in the `LICENSE` file.

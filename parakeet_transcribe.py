# parakeet_transcribe.py
#
# Description:
# This script transcribes an audio file using NVIDIA Parakeet TDT 0.6B.
# - It takes a file path as input, assuming pre-processed audio (16kHz mono).
# - Calls ASRModel.transcribe() with timestamps=True and return_hypotheses=True.
# - Correctly handles the Hypothesis object returned by the ASR model.
# - Uses the 'start' and 'end' keys from the Hypothesis's timestamp dictionaries,
#   which are already in seconds.
# - Accepts an --audio_start_offset argument to adjust all timestamps globally.
# - Accepts a --force_float32 argument to run the model in float32 precision
#   on GPU, overriding automatic mixed precision.
# - Writes an error SRT if CUDA OOM or other critical errors occur, ensuring
#   the error message is flushed to the file.
# - Supports generation of SRT files based on segment-level or word-level timestamps.

import nemo.collections.asr as nemo_asr
import sys
import os
import soundfile as sf
import librosa
import argparse
import numpy as np
import torch
import gc
from pathlib import Path
from torch.cuda.amp import autocast

# Ensure repository modules are importable when launched without a
# preconfigured PYTHONPATH. Python adds the script directory to
# ``sys.path`` by default when executing a file directly, but inserting it
# explicitly guards against edge cases and mirrors the behavior of the
# separation script.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from srt_utils import format_time_srt, postprocess_segments, write_srt, normalize_text


def _audit(events_in, events_out):
    tin = normalize_text(" ".join(e["text"] for e in events_in))
    tout = normalize_text(" ".join(e["text"] for e in events_out))
    if len(tout) < len(tin) * 0.98:
        print(
            "[WARN] Postprocess lost text:",
            f"in={len(tin)} out={len(tout)} Δ={len(tin) - len(tout)}",
        )

def generate_srt_from_processed_timestamps(
    all_processed_timestamps: list,
    srt_file_path: str,
    timestamp_type: str = "segment",
    max_words: int = 12,
    max_duration: float = 6.0,
    pause: float = 0.6,
):
    """
    Generates an SRT file from a list of processed timestamp dictionaries.

    This function can create SRT files based on either 'segment' level timestamps
    or 'word' level timestamps. If no timestamps are provided, it writes a
    minimal SRT file indicating that no data was processed or an error occurred.

    Args:
        all_processed_timestamps (list): A list of dictionaries, where each
            dictionary contains timestamp data.
            For 'segment' type, each dict should have 'text_from_model',
            'start_seconds', and 'end_seconds'.
            For 'word' type, each dict should have 'word_from_model',
            'start_seconds', and 'end_seconds'.
        srt_file_path (str): The full path to the SRT file to be created/overwritten.
        timestamp_type (str, optional): The type of timestamps being processed.
            Can be "segment" or "word". Defaults to "segment".

    Raises:
        IOError: If there's an error writing to the SRT file.
    """
    if not all_processed_timestamps:
        print(f"Warning: No {timestamp_type} timestamps provided to generate SRT.", file=sys.stderr)
        # Attempt to write a minimal SRT indicating no timestamps were processed
        try:
            with open(srt_file_path, 'w', encoding='utf-8') as f_empty:
                f_empty.write(f"1\n00:00:00,000 --> 00:00:01,000\n[No {timestamp_type}s found or error in processing]\n\n")
                f_empty.flush()
                os.fsync(f_empty.fileno()) # Ensure it's written to disk
        except Exception as e_write:
            print(f"Error writing empty/error SRT: {e_write}", file=sys.stderr)
        return

    with open(srt_file_path, 'w', encoding='utf-8') as f:
        if timestamp_type == "segment":
            for i, segment_data in enumerate(all_processed_timestamps):
                segment_num = i + 1
                text = segment_data.get('text_from_model', "[No segment text]").strip()
                start_s = segment_data.get('start_seconds', 0.0)
                end_s = segment_data.get('end_seconds', start_s + 0.1) # Ensure end is after start
                if not text: continue # Skip empty segments
                f.write(f"{segment_num}\n")
                f.write(f"{format_time_srt(start_s)} --> {format_time_srt(end_s)}\n")
                f.write(text + "\n\n")
            print(f"SRT file generated from {len(all_processed_timestamps)} segments at '{srt_file_path}'", file=sys.stderr)

        elif timestamp_type == "word":
            segment_num = 1
            current_words = []
            segment_start = None
            last_end = 0.0

            def flush(final_end: float):
                nonlocal segment_num, current_words, segment_start
                if not current_words:
                    return
                f.write(f"{segment_num}\n")
                f.write(f"{format_time_srt(segment_start)} --> {format_time_srt(final_end)}\n")
                f.write(" ".join(current_words).strip() + "\n\n")
                segment_num += 1
                current_words = []
                segment_start = None

            for item in all_processed_timestamps:
                word_val = item.get("word_from_model", "[unknown]")
                start_s = item.get("start_seconds")
                end_s = item.get("end_seconds")
                if start_s is None or end_s is None:
                    continue

                if not current_words:
                    segment_start = start_s
                    current_words.append(str(word_val))
                else:
                    gap = start_s - last_end
                    duration = end_s - segment_start
                    if gap >= pause or len(current_words) >= max_words or duration > max_duration:
                        # Prefer breaking at punctuation if present
                        punct_idx = None
                        for idx in range(len(current_words) - 1, -1, -1):
                            if current_words[idx].endswith((",", ".", "!", "?", ";", ":")):
                                punct_idx = idx
                                break
                        flush(last_end)
                        segment_start = start_s
                        current_words = [str(word_val)]
                    else:
                        current_words.append(str(word_val))
                last_end = end_s

            if current_words:
                flush(last_end)
            print(
                f"SRT file generated by grouping {len(all_processed_timestamps)} words at '{srt_file_path}'",
                file=sys.stderr,
            )
        else:
            print(f"Error: Unknown timestamp_type '{timestamp_type}' for SRT generation.", file=sys.stderr)
        f.flush() # Ensure data is written to buffer
        os.fsync(f.fileno()) # Ensure data is written to disk from buffer

# --- Main Transcription Logic ---
def main():
    """
    Main function to perform audio transcription using NVIDIA Parakeet ASR model.

    Parses command-line arguments, loads the ASR model, transcribes the audio file,
    processes timestamps, and generates an SRT subtitle file. Handles errors
    robustly, including CUDA OutOfMemoryError, and writes error SRT files.

    Command-line Arguments:
        audio_file_path (str): Path to the input audio file (e.g., WAV).
                               Expected to be 16kHz mono.
        srt_output_file_path (str): Path to save the generated SRT file.
        --audio_start_offset (float, optional): Global start time offset in seconds
                                                to apply to all timestamps. Defaults to 0.0.
        --force_float32 (bool, optional): Force the model to run in float32 precision
                                          on GPU, even if bfloat16/float16 is available.
                                          Defaults to False.

    Raises:
        SystemExit: If critical errors occur (e.g., file not found, transcription failure)
                    or if CUDA OOM occurs.
        torch.cuda.OutOfMemoryError: If the GPU runs out of memory during model loading or transcription.
        Exception: For other unexpected errors during the process.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using NVIDIA Parakeet TDT 0.6B. Assumes input audio is pre-processed (16kHz mono). Uses direct timestamp values from model as seconds.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("audio_file_path", type=str, help="Path to the input audio file (e.g., WAV). Expected to be 16kHz mono.")
    parser.add_argument("srt_output_file_path", type=str, help="Path to save the generated SRT file.")
    parser.add_argument("--audio_start_offset", type=float, default=0.0, help="Global start time offset in seconds from the original media to apply to all timestamps.")
    parser.add_argument("--force_float32", action="store_true", help="Force the model to run in float32 precision on GPU, even if bfloat16/float16 is available.")
    parser.add_argument("--segmenter", choices=["word", "segment"], default="word", help="Timestamp type to group for SRT generation")
    parser.add_argument("--max_words", type=int, default=12, help="Maximum words per subtitle when using word segmentation")
    parser.add_argument("--max_duration", type=float, default=6.0, help="Maximum subtitle duration in seconds for word segmentation")
    parser.add_argument("--pause", type=float, default=0.6, help="Inter-word pause (s) that triggers a new subtitle when using word segmentation")
    parser.add_argument("--fps", type=float, default=24.0, help="Video FPS for frame snapping")
    parser.add_argument("--max_chars_per_line", type=int, default=40)
    parser.add_argument("--pause_ms", type=int, default=220, help="Minimum inter-word silence to open a split candidate")
    parser.add_argument("--cps", type=float, default=20.0, help="Target characters-per-second reading speed")
    parser.add_argument("--no_spacy", action="store_true", help="Disable spaCy hints even if available")
    args = parser.parse_args()

    audio_path = args.audio_file_path
    srt_path = args.srt_output_file_path
    global_offset_seconds = args.audio_start_offset
    force_float32 = args.force_float32
    segmenter = args.segmenter
    max_words = args.max_words
    max_duration = args.max_duration
    pause_threshold = args.pause
    fps = args.fps
    max_chars_per_line = args.max_chars_per_line
    pause_ms = args.pause_ms
    cps = args.cps
    use_spacy = not args.no_spacy

    # Define a helper function to write error SRTs immediately
    def write_error_srt(message: str):
        """Writes a standardized error message to the SRT file and flushes it."""
        try:
            with open(srt_path, 'w', encoding='utf-8') as f_err:
                # Create a minimal SRT with the error message
                f_err.write(f"1\n00:00:00,000 --> 00:00:01,000\n[{message} for {os.path.basename(audio_path)}]\n\n")
                f_err.flush() # Ensure the error message is written out
                os.fsync(f_err.fileno()) # Force write to disk
            print(f"Error SRT written: {message}", file=sys.stderr)
        except Exception as e_write:
            # This is a critical failure if even the error SRT cannot be written
            print(f"Critical error: Could not write error SRT: {e_write}", file=sys.stderr)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'", file=sys.stderr)
        write_error_srt(f"Audio file not found: {os.path.basename(audio_path)}")
        sys.exit(1)

    asr_model = None
    long_audio_settings_applied = False # Flag to track if long audio settings were changed
    use_cuda = torch.cuda.is_available()
    audio_duration_seconds = 0.0

    try:
        model_name = "nvidia/parakeet-tdt-0.6b-v2" # Specify the Parakeet model
        print(f"Loading ASR model '{model_name}'...", file=sys.stderr)
        # Load the ASR model from NeMo's pre-trained models
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, strict=False)

        model_dtype = torch.float32 # Default to float32
        if use_cuda:
            print("CUDA is available. Moving model to GPU.", file=sys.stderr)
            asr_model = asr_model.cuda()
            if force_float32:
                asr_model = asr_model.to(dtype=torch.float32)
                model_dtype = torch.float32
                print("Model forced to float32 precision on GPU.", file=sys.stderr)
            else:
                # Attempt to use bfloat16 or float16 for better performance if available
                try:
                    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                        asr_model = asr_model.to(dtype=torch.bfloat16)
                        model_dtype = torch.bfloat16
                        print("Model explicitly converted to bfloat16 precision.", file=sys.stderr)
                    else:
                        asr_model = asr_model.half() # .half() corresponds to float16
                        model_dtype = torch.float16
                        print("bfloat16 not supported, model converted to float16 precision.", file=sys.stderr)
                except Exception as e_prec:
                    print(f"Warning: Could not convert model to bfloat16/float16: {e_prec}. Using float32 on GPU.", file=sys.stderr)
                    asr_model = asr_model.to(dtype=torch.float32) # Fallback to float32
                    model_dtype = torch.float32
        else:
            print("CUDA not available. Using CPU (float32).", file=sys.stderr)
        
        asr_model.eval() # Set model to evaluation mode

        # Get model's expected sample rate
        target_sr_from_model_cfg = asr_model.cfg.preprocessor.sample_rate
        print(f"Model expects sample rate: {target_sr_from_model_cfg} Hz. Input audio should match this.", file=sys.stderr)

        print(f"Getting audio info for: '{audio_path}' (expected to be pre-processed)...", file=sys.stderr)
        try:
            # Use soundfile to get audio information
            info = sf.info(audio_path)
            audio_duration_seconds = info.duration
            actual_sr = info.samplerate
            actual_channels = info.channels
            print(f"Input audio duration: {audio_duration_seconds:.2f}s, Sample rate: {actual_sr}Hz, Channels: {actual_channels}", file=sys.stderr)
            # Crucial checks for audio properties
            if actual_sr != target_sr_from_model_cfg:
                  print(f"CRITICAL WARNING: Input audio sample rate ({actual_sr}Hz) differs from model's expected rate ({target_sr_from_model_cfg}Hz). Results may be suboptimal.", file=sys.stderr)
            if actual_channels != 1:
                  print(f"CRITICAL WARNING: Input audio has {actual_channels} channels. Model expects mono (1 channel). Results may be suboptimal.", file=sys.stderr)
        except Exception as e_info:
            print(f"Warning: Could not get audio info using soundfile: {e_info}. Attempting with librosa for duration.", file=sys.stderr)
            try:
                # Fallback to librosa for duration if soundfile fails
                audio_duration_seconds = librosa.get_duration(path=audio_path)
                print(f"Input audio duration (via librosa): {audio_duration_seconds:.2f}s", file=sys.stderr)
            except Exception as e_lib_dur:
                err_msg = f"Could not determine audio duration: {e_lib_dur}"
                print(f"Error: {err_msg}", file=sys.stderr)
                write_error_srt(err_msg)
                sys.exit(1)

        # Configure model for long audio if duration exceeds threshold
        LONG_AUDIO_THRESHOLD_S = 480 # 8 minutes
        if audio_duration_seconds > LONG_AUDIO_THRESHOLD_S:
            try:
                print(f"Audio duration ({audio_duration_seconds:.2f}s) > {LONG_AUDIO_THRESHOLD_S}s. Applying long audio settings.", file=sys.stderr)
                # Change attention mechanism and subsampling for longer audio files
                asr_model.change_attention_model("rel_pos_local_attn", [256,256])
                asr_model.change_subsampling_conv_chunking_factor(1)
                long_audio_settings_applied = True
                print("Long audio settings applied: Local Attention and Auto Conv Chunking.", file=sys.stderr)
            except Exception as setting_e:
                print(f"Warning: Failed to apply long audio settings: {setting_e}. Proceeding without them.", file=sys.stderr)
        
        print(f"Starting transcription for '{audio_path}' (using file path input)...", file=sys.stderr)
        
        transcribe_input_files = [audio_path] # Model expects a list of file paths
        # Determine if automatic mixed precision (AMP) should be used with autocast
        use_amp_autocast = use_cuda and not force_float32 and model_dtype != torch.float32
        
        # Perform transcription within autocast context if using mixed precision
        with autocast(dtype=model_dtype if use_amp_autocast else torch.float32, enabled=use_amp_autocast):
              print(f"Transcribing with precision: {model_dtype if use_cuda else 'float32 (CPU)'}. Autocast enabled: {use_amp_autocast}", file=sys.stderr)
              # Call the transcribe method with timestamp and hypothesis options
              output_from_transcribe = asr_model.transcribe(transcribe_input_files, timestamps=True, return_hypotheses=True)

        if not output_from_transcribe or not isinstance(output_from_transcribe, list) or not output_from_transcribe[0]:
            err_msg = "Transcription failed or produced no hypotheses"
            print(f"Error: {err_msg}", file=sys.stderr)
            write_error_srt(err_msg)
            sys.exit(1)
        
        # The result for a single file is the first element of the list
        first_result = output_from_transcribe[0]
        full_transcript = None
        
        # Check if the output is a NeMo Hypothesis object and extract data
        if hasattr(first_result, 'text') and hasattr(first_result, 'timestamp') and first_result.timestamp is not None:
            print("Processing output as NeMo Hypothesis object.", file=sys.stderr)
            hypothesis = first_result
            full_transcript = hypothesis.text
            # Get segment-level and word-level timestamps if available
            segment_timestamps_from_model = hypothesis.timestamp.get('segment', [])
            word_timestamps_from_model = hypothesis.timestamp.get('word', [])
        else:
            err_msg = "Transcription output format not recognized or timestamp data missing"
            print(f"Error: {err_msg}", file=sys.stderr)
            # Debugging information about the received output
            print(f"Type of output_from_transcribe[0]: {type(first_result)}", file=sys.stderr)
            if hasattr(first_result, '__dict__'): print(f"Vars: {vars(first_result)}", file=sys.stderr)
            write_error_srt(err_msg)
            sys.exit(1)
            
        print(f"\nFull Transcript:\n{full_transcript}\n", file=sys.stderr)

        # Build segments for post-processing
        segments = []
        if segment_timestamps_from_model:
            for seg in segment_timestamps_from_model:
                text_val = seg.get("segment")
                start_s = seg.get("start")
                end_s = seg.get("end")
                if start_s is None or end_s is None or text_val is None:
                    continue
                words = []
                if word_timestamps_from_model:
                    for w in word_timestamps_from_model:
                        ws = w.get("start")
                        we = w.get("end")
                        word = w.get("word")
                        if ws is None or we is None or word is None:
                            continue
                        if ws >= start_s and we <= end_s:
                            words.append({"word": str(word), "start": float(ws), "end": float(we)})
                segments.append({"start": float(start_s), "end": float(end_s), "text": str(text_val), "words": words})
        elif word_timestamps_from_model:
            words_list = []
            for w in word_timestamps_from_model:
                ws = w.get("start")
                we = w.get("end")
                word = w.get("word")
                if ws is None or we is None or word is None:
                    continue
                words_list.append({"word": str(word), "start": float(ws), "end": float(we)})
            if words_list:
                text = " ".join(w["word"] for w in words_list)
                segments.append({"start": words_list[0]["start"], "end": words_list[-1]["end"], "text": text, "words": words_list})
        elif full_transcript is not None:
            segments.append({"start": 0.0, "end": audio_duration_seconds, "text": full_transcript})

        if not segments:
            write_error_srt("No transcript text available")
            return

        if global_offset_seconds != 0.0:
            print(f"Applying global start offset of {global_offset_seconds:.3f} seconds to all timestamps.", file=sys.stderr)
            for seg in segments:
                seg["start"] += global_offset_seconds
                seg["end"] += global_offset_seconds
                if seg.get("words"):
                    for w in seg["words"]:
                        w["start"] += global_offset_seconds
                        w["end"] += global_offset_seconds

        processed = postprocess_segments(
            segments,
            max_chars_per_line=max_chars_per_line,
            max_lines=2,
            pause_ms=pause_ms,
            cps_target=cps,
            snap_fps=fps,
            use_spacy=use_spacy,
        )
        _audit(segments, processed)
        write_srt(processed, srt_path)
        print(f"SRT file generated at '{srt_path}'", file=sys.stderr)

        print(f"SRT file processing completed for '{srt_path}'", file=sys.stderr)

    except torch.cuda.OutOfMemoryError as e_oom:
        err_msg = f"CUDA OutOfMemoryError: {e_oom}"
        print(err_msg, file=sys.stderr)
        print("The audio file might be too long for your GPU's VRAM even with long audio settings, or the batch size is too large for the model on this GPU.", file=sys.stderr)
        print("Try reducing batch size if applicable, or using a machine with more VRAM, or processing shorter audio segments.", file=sys.stderr)
        write_error_srt("CUDA OutOfMemoryError") # Write error to SRT
        sys.exit(1) 
    except SystemExit: # Allow sys.exit to propagate cleanly
        raise
    except Exception as e:
        err_msg = f"An unexpected error occurred: {e}"
        print(err_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
        write_error_srt(f"Error during transcription: {str(e)[:100]}") # Write concise error to SRT
        sys.exit(1)
    finally:
        # Cleanup: revert model settings, move model to CPU, and clear cache
        if asr_model is not None:
            if long_audio_settings_applied:
                try:
                    print("Reverting long audio settings...", file=sys.stderr)
                    # Revert to default attention and subsampling settings
                    asr_model.change_attention_model(self_attention_model="rel_pos") # Default for Parakeet
                    asr_model.change_subsampling_conv_chunking_factor(-1) # Auto, default
                    print("Long audio settings reverted.", file=sys.stderr)
                except Exception as revert_e:
                    print(f"Warning: Failed to revert long audio settings: {revert_e}", file=sys.stderr)
            try:
                # Move model to CPU and clear memory
                if hasattr(asr_model, 'cpu'): asr_model.cpu()
                del asr_model
                gc.collect() # Force garbage collection
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Clear CUDA cache
                print("Model moved to CPU and CUDA cache cleared (if applicable).", file=sys.stderr)
            except Exception as cleanup_e:
                print(f"Error during model cleanup: {cleanup_e}", file=sys.stderr)

if __name__ == "__main__":
    main()

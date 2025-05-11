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
from torch.cuda.amp import autocast

# --- SRT Formatting Function ---
def format_time_srt(seconds: float) -> str:
    """
    Formats a duration in seconds into the SRT (SubRip Text) time format.

    The format is HH:MM:SS,mmm (hours, minutes, seconds, milliseconds).
    If the input is not a valid number or is negative, it defaults to 0.0 seconds.

    Args:
        seconds (float): The time in seconds to format.

    Returns:
        str: The time formatted as an SRT timestamp string (e.g., "00:01:23,456").
    """
    if not isinstance(seconds, (int, float)):
        seconds = 0.0 # Default to 0 if input is not a number
    if seconds < 0:
        seconds = 0.0 # Treat negative times as 0
    total_seconds_int = int(seconds)
    milliseconds = int(round((seconds - total_seconds_int) * 1000))
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt_from_processed_timestamps(all_processed_timestamps: list, srt_file_path: str, timestamp_type: str = "segment"):
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
            # Logic for grouping words into SRT segments
            segment_num = 1
            current_segment_words = []
            segment_start_time_s = None
            last_word_end_time_s = 0.0
            MAX_WORDS_PER_SEGMENT = 12    # Maximum words allowed in a single SRT entry
            MAX_DURATION_SECONDS = 6.0    # Maximum duration of a single SRT entry
            PAUSE_THRESHOLD_SECONDS = 0.7 # Pause duration to force a new segment

            for i, item in enumerate(all_processed_timestamps):
                word_val = item.get('word_from_model', '[unknown]')
                start_s = item.get('start_seconds')
                end_s = item.get('end_seconds')

                if start_s is None or end_s is None: continue # Skip if timing is missing

                if not current_segment_words: # First word of a new segment
                    segment_start_time_s = start_s
                    current_segment_words.append(str(word_val))
                    last_word_end_time_s = end_s
                else:
                    force_new_segment = False
                    # Check conditions to start a new segment
                    if (start_s - last_word_end_time_s) >= PAUSE_THRESHOLD_SECONDS: force_new_segment = True
                    elif len(current_segment_words) >= MAX_WORDS_PER_SEGMENT: force_new_segment = True
                    elif (end_s - segment_start_time_s) > MAX_DURATION_SECONDS: force_new_segment = True

                    if force_new_segment:
                        # Write current segment to file
                        f.write(f"{segment_num}\n")
                        f.write(f"{format_time_srt(segment_start_time_s)} --> {format_time_srt(last_word_end_time_s)}\n")
                        f.write(" ".join(current_segment_words).strip() + "\n\n")
                        segment_num += 1
                        # Start new segment with current word
                        current_segment_words = [str(word_val)]
                        segment_start_time_s = start_s
                    else:
                        # Add word to current segment
                        current_segment_words.append(str(word_val))
                last_word_end_time_s = end_s # Update the end time of the last word processed

            # Write any remaining words in the last segment
            if current_segment_words:
                f.write(f"{segment_num}\n")
                f.write(f"{format_time_srt(segment_start_time_s)} --> {format_time_srt(last_word_end_time_s)}\n")
                f.write(" ".join(current_segment_words).strip() + "\n\n")
            print(f"SRT file generated by grouping {len(all_processed_timestamps)} words at '{srt_file_path}'", file=sys.stderr)
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
    args = parser.parse_args()

    audio_path = args.audio_file_path
    srt_path = args.srt_output_file_path
    global_offset_seconds = args.audio_start_offset
    force_float32 = args.force_float32

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

        final_timestamps_for_srt = []
        srt_generation_type = "none" # To track which type of SRT will be generated

        # Process segment timestamps if available
        if segment_timestamps_from_model:
            srt_generation_type = "segment"
            print(f"Found {len(segment_timestamps_from_model)} segment timestamps. Using 'start' and 'end' keys (expected in seconds).", file=sys.stderr)
            for idx, ts_item in enumerate(segment_timestamps_from_model):
                text_val = ts_item.get('segment') # Key for segment text
                start_s_seconds_raw = ts_item.get('start') # Key for start time in seconds
                end_s_seconds_raw = ts_item.get('end')     # Key for end time in seconds
                
                if start_s_seconds_raw is not None and end_s_seconds_raw is not None and text_val is not None:
                    try:
                        # Convert to float and store
                        final_timestamps_for_srt.append({
                            'text_from_model': str(text_val),
                            'start_seconds': float(start_s_seconds_raw),
                            'end_seconds': float(end_s_seconds_raw)
                        })
                    except ValueError as e:
                        print(f"Debug: Could not convert segment offsets to float: start='{start_s_seconds_raw}', end='{end_s_seconds_raw}'. Error: {e}", file=sys.stderr)
                        continue # Skip this malformed segment
                else:
                    print(f"Debug: Malformed segment timestamp item (missing 'start', 'end', or 'segment' text key): {ts_item}", file=sys.stderr)
        
        # Fallback to word timestamps if segment timestamps were not available or processed
        if not final_timestamps_for_srt and word_timestamps_from_model:
            srt_generation_type = "word"
            print(f"Found {len(word_timestamps_from_model)} word timestamps. Using 'start_time', 'end_time', 'word' keys.", file=sys.stderr)
            for idx, ts_item in enumerate(word_timestamps_from_model):
                word_val = ts_item.get('word') # Key for the word
                # Note: Parakeet TDT 0.6B directly gives 'start' and 'end' in seconds for words too.
                start_s_seconds_raw = ts_item.get('start')
                end_s_seconds_raw = ts_item.get('end')

                if start_s_seconds_raw is not None and end_s_seconds_raw is not None and word_val is not None:
                    try:
                        final_timestamps_for_srt.append({
                            'word_from_model': str(word_val),
                            'start_seconds': float(start_s_seconds_raw),
                            'end_seconds': float(end_s_seconds_raw)
                        })
                    except ValueError as e:
                        print(f"Debug: Could not convert word offsets to float: start='{start_s_seconds_raw}', end='{end_s_seconds_raw}'. Error: {e}", file=sys.stderr)
                        continue
                else:
                    print(f"Debug: Malformed word timestamp item (missing 'start', 'end', or 'word'): {ts_item}", file=sys.stderr)

        # Apply global start time offset if specified
        if global_offset_seconds != 0.0 and final_timestamps_for_srt:
            print(f"Applying global start offset of {global_offset_seconds:.3f} seconds to all timestamps.", file=sys.stderr)
            for ts_data in final_timestamps_for_srt:
                ts_data['start_seconds'] += global_offset_seconds
                ts_data['end_seconds'] += global_offset_seconds
                # Ensure timestamps are not negative after offset
                if ts_data['start_seconds'] < 0: ts_data['start_seconds'] = 0.0
                if ts_data['end_seconds'] < 0: ts_data['end_seconds'] = 0.01 # Ensure end is slightly after start if both become negative
                # Ensure end time is always after start time
                if ts_data['end_seconds'] <= ts_data['start_seconds']:
                    ts_data['end_seconds'] = ts_data['start_seconds'] + 0.1 # Minimal duration


        # Generate SRT file based on processed timestamps
        if final_timestamps_for_srt:
            if srt_generation_type == "segment":
                generate_srt_from_processed_timestamps(final_timestamps_for_srt, srt_path, timestamp_type="segment")
            elif srt_generation_type == "word":
                generate_srt_from_processed_timestamps(final_timestamps_for_srt, srt_path, timestamp_type="word")
        else:
            # Fallback if no detailed timestamps could be processed, but we have a full transcript
            err_msg = "No segment or word timestamps processed"
            print(f"Error: {err_msg}. Cannot generate detailed SRT.", file=sys.stderr)
            if full_transcript is not None:
                # Create a single SRT entry spanning the whole audio duration (with offset)
                print(f"Writing fallback SRT with full transcript and audio duration.", file=sys.stderr)
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write("1\n")
                    adjusted_start_time_full = global_offset_seconds
                    adjusted_end_time_full = audio_duration_seconds + global_offset_seconds
                    # Sanitize times
                    if adjusted_start_time_full < 0: adjusted_start_time_full = 0.0
                    if adjusted_end_time_full < adjusted_start_time_full:
                        adjusted_end_time_full = adjusted_start_time_full + max(1.0, audio_duration_seconds) # Use audio duration or 1s
                    if adjusted_end_time_full <= adjusted_start_time_full: adjusted_end_time_full = adjusted_start_time_full + 0.1

                    f.write(f"{format_time_srt(adjusted_start_time_full)} --> {format_time_srt(adjusted_end_time_full)}\n")
                    f.write(full_transcript.strip() + "\n\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(f"SRT file generated with full transcript at '{srt_path}'. Offset applied if any.", file=sys.stderr)
            else:
                 # If even full_transcript is None, write an error SRT
                write_error_srt("No transcript text available")

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

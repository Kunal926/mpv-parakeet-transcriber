# parakeet_transcribe.py (v17f - Robust Error SRT)
#
# Description:
# This script transcribes an audio file using NVIDIA Parakeet TDT 0.6B.
# - It takes a file path as input, assuming pre-processed audio.
# - Calls ASRModel.transcribe() with timestamps=True and return_hypotheses=True.
# - Correctly handles the Hypothesis object returned.
# - Uses the 'start' and 'end' keys from the Hypothesis's timestamp dictionaries
#   (which are already in seconds).
# - Accepts an --audio_start_offset argument to adjust all timestamps.
# - Accepts a --force_float32 argument to run model in float32 on GPU.
# - Writes an error SRT if CUDA OOM or other critical errors occur, with flushing.

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
    if not isinstance(seconds, (int, float)):
        seconds = 0.0
    if seconds < 0:
        seconds = 0.0
    total_seconds_int = int(seconds)
    milliseconds = int(round((seconds - total_seconds_int) * 1000))
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt_from_processed_timestamps(all_processed_timestamps: list, srt_file_path: str, timestamp_type: str = "segment"):
    if not all_processed_timestamps:
        print(f"Warning: No {timestamp_type} timestamps provided to generate SRT.", file=sys.stderr)
        # Attempt to write a minimal SRT indicating no timestamps were processed
        try:
            with open(srt_file_path, 'w', encoding='utf-8') as f_empty:
                f_empty.write(f"1\n00:00:00,000 --> 00:00:01,000\n[No {timestamp_type}s found or error in processing]\n\n")
                f_empty.flush()
                os.fsync(f_empty.fileno())
        except Exception as e_write:
            print(f"Error writing empty/error SRT: {e_write}", file=sys.stderr)
        return

    with open(srt_file_path, 'w', encoding='utf-8') as f:
        if timestamp_type == "segment":
            for i, segment_data in enumerate(all_processed_timestamps):
                segment_num = i + 1
                text = segment_data.get('text_from_model', "[No segment text]").strip()
                start_s = segment_data.get('start_seconds', 0.0)
                end_s = segment_data.get('end_seconds', start_s + 0.1)
                if not text: continue
                f.write(f"{segment_num}\n")
                f.write(f"{format_time_srt(start_s)} --> {format_time_srt(end_s)}\n")
                f.write(text + "\n\n")
            print(f"SRT file generated from {len(all_processed_timestamps)} segments at '{srt_file_path}'", file=sys.stderr)

        elif timestamp_type == "word":
            # ... (word processing logic from v17d, no changes needed here for robustness) ...
            segment_num = 1
            current_segment_words = []
            segment_start_time_s = None
            last_word_end_time_s = 0.0
            MAX_WORDS_PER_SEGMENT = 12
            MAX_DURATION_SECONDS = 6.0
            PAUSE_THRESHOLD_SECONDS = 0.7
            for i, item in enumerate(all_processed_timestamps):
                word_val = item.get('word_from_model', '[unknown]')
                start_s = item.get('start_seconds')
                end_s = item.get('end_seconds')
                if start_s is None or end_s is None: continue
                if not current_segment_words:
                    segment_start_time_s = start_s
                    current_segment_words.append(str(word_val))
                    last_word_end_time_s = end_s
                else:
                    force_new_segment = False
                    if (start_s - last_word_end_time_s) >= PAUSE_THRESHOLD_SECONDS: force_new_segment = True
                    elif len(current_segment_words) >= MAX_WORDS_PER_SEGMENT: force_new_segment = True
                    elif (end_s - segment_start_time_s) > MAX_DURATION_SECONDS: force_new_segment = True
                    if force_new_segment:
                        f.write(f"{segment_num}\n")
                        f.write(f"{format_time_srt(segment_start_time_s)} --> {format_time_srt(last_word_end_time_s)}\n")
                        f.write(" ".join(current_segment_words).strip() + "\n\n")
                        segment_num += 1
                        current_segment_words = [str(word_val)]
                        segment_start_time_s = start_s
                    else:
                        current_segment_words.append(str(word_val))
                last_word_end_time_s = end_s
                if i == len(all_processed_timestamps) - 1 and current_segment_words:
                    f.write(f"{segment_num}\n")
                    f.write(f"{format_time_srt(segment_start_time_s)} --> {format_time_srt(last_word_end_time_s)}\n")
                    f.write(" ".join(current_segment_words).strip() + "\n\n")
            print(f"SRT file generated by grouping {len(all_processed_timestamps)} words at '{srt_file_path}'", file=sys.stderr)
        else:
            print(f"Error: Unknown timestamp_type '{timestamp_type}' for SRT generation.", file=sys.stderr)
        f.flush() # Ensure data is written
        os.fsync(f.fileno()) # Ensure it's written to disk

# --- Main Transcription Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using NVIDIA Parakeet TDT 0.6B. Assumes input audio is pre-processed. Uses direct timestamp values from model as seconds.",
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

    # Define a helper to write error SRTs
    def write_error_srt(message):
        try:
            with open(srt_path, 'w', encoding='utf-8') as f_err:
                f_err.write(f"1\n00:00:00,000 --> 00:00:01,000\n[{message} for {os.path.basename(audio_path)}]\n\n")
                f_err.flush()
                os.fsync(f_err.fileno())
            print(f"Error SRT written: {message}", file=sys.stderr)
        except Exception as e_write:
            print(f"Critical error: Could not write error SRT: {e_write}", file=sys.stderr)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'", file=sys.stderr)
        write_error_srt(f"Audio file not found {audio_path}")
        sys.exit(1)

    asr_model = None
    long_audio_settings_applied = False
    use_cuda = torch.cuda.is_available()
    audio_duration_seconds = 0.0

    try:
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        print(f"Loading ASR model '{model_name}'...", file=sys.stderr)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, strict=False)

        model_dtype = torch.float32
        if use_cuda:
            print("CUDA is available. Moving model to GPU.", file=sys.stderr)
            asr_model = asr_model.cuda()
            if force_float32:
                asr_model = asr_model.to(dtype=torch.float32)
                model_dtype = torch.float32
                print("Model forced to float32 precision on GPU.", file=sys.stderr)
            else:
                try:
                    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                        asr_model = asr_model.to(dtype=torch.bfloat16)
                        model_dtype = torch.bfloat16
                        print("Model explicitly converted to bfloat16 precision.", file=sys.stderr)
                    else:
                        asr_model = asr_model.half()
                        model_dtype = torch.float16
                        print("bfloat16 not supported, model converted to float16 precision.", file=sys.stderr)
                except Exception as e_prec:
                    print(f"Warning: Could not convert model to bfloat16/float16: {e_prec}. Using float32 on GPU.", file=sys.stderr)
                    asr_model = asr_model.to(dtype=torch.float32)
                    model_dtype = torch.float32
        else:
            print("CUDA not available. Using CPU (float32).", file=sys.stderr)
        
        asr_model.eval()

        target_sr_from_model_cfg = asr_model.cfg.preprocessor.sample_rate
        print(f"Model expects sample rate: {target_sr_from_model_cfg} Hz. Input audio should match this.", file=sys.stderr)

        print(f"Getting audio info for: '{audio_path}' (expected to be pre-processed)...", file=sys.stderr)
        try:
            info = sf.info(audio_path)
            audio_duration_seconds = info.duration
            actual_sr = info.samplerate
            actual_channels = info.channels
            print(f"Input audio duration: {audio_duration_seconds:.2f}s, Sample rate: {actual_sr}Hz, Channels: {actual_channels}", file=sys.stderr)
            if actual_sr != target_sr_from_model_cfg:
                 print(f"CRITICAL WARNING: Input audio sample rate ({actual_sr}Hz) differs from model's expected rate ({target_sr_from_model_cfg}Hz).", file=sys.stderr)
            if actual_channels != 1:
                 print(f"CRITICAL WARNING: Input audio has {actual_channels} channels. Model expects mono (1 channel).", file=sys.stderr)
        except Exception as e_info:
            print(f"Warning: Could not get audio info using soundfile: {e_info}. Attempting with librosa.", file=sys.stderr)
            try:
                audio_duration_seconds = librosa.get_duration(path=audio_path)
                print(f"Input audio duration (via librosa): {audio_duration_seconds:.2f}s", file=sys.stderr)
            except Exception as e_lib_dur:
                err_msg = f"Could not determine audio duration: {e_lib_dur}"
                print(f"Error: {err_msg}", file=sys.stderr)
                write_error_srt(err_msg)
                sys.exit(1)

        LONG_AUDIO_THRESHOLD_S = 480
        if audio_duration_seconds > LONG_AUDIO_THRESHOLD_S:
            try:
                print(f"Audio duration ({audio_duration_seconds:.2f}s) > {LONG_AUDIO_THRESHOLD_S}s. Applying long audio settings.", file=sys.stderr)
                asr_model.change_attention_model("rel_pos_local_attn", [256,256])
                asr_model.change_subsampling_conv_chunking_factor(1)
                long_audio_settings_applied = True
                print("Long audio settings applied: Local Attention and Auto Conv Chunking.", file=sys.stderr)
            except Exception as setting_e:
                print(f"Warning: Failed to apply long audio settings: {setting_e}. Proceeding without them.", file=sys.stderr)
        
        print(f"Starting transcription for '{audio_path}' (using file path input)...", file=sys.stderr)
        
        transcribe_input_files = [audio_path]
        use_amp_autocast = use_cuda and not force_float32 and model_dtype != torch.float32
        
        with autocast(dtype=model_dtype if use_amp_autocast else torch.float32, enabled=use_amp_autocast):
             print(f"Transcribing with precision: {model_dtype if use_cuda else 'float32 (CPU)'}. Autocast enabled: {use_amp_autocast}", file=sys.stderr)
             output_from_transcribe = asr_model.transcribe(transcribe_input_files, timestamps=True, return_hypotheses=True)

        if not output_from_transcribe or not isinstance(output_from_transcribe, list) or not output_from_transcribe[0]:
            err_msg = "Transcription failed or produced no hypotheses"
            print(f"Error: {err_msg}", file=sys.stderr)
            write_error_srt(err_msg)
            sys.exit(1)
        
        first_result = output_from_transcribe[0]
        full_transcript = None
        
        if hasattr(first_result, 'text') and hasattr(first_result, 'timestamp') and first_result.timestamp is not None:
            print("Processing output as NeMo Hypothesis object.", file=sys.stderr)
            hypothesis = first_result
            full_transcript = hypothesis.text
            segment_timestamps_from_model = hypothesis.timestamp.get('segment', [])
            word_timestamps_from_model = hypothesis.timestamp.get('word', [])
        else:
            err_msg = "Transcription output format not recognized"
            print(f"Error: {err_msg}", file=sys.stderr)
            # ... (debug prints) ...
            write_error_srt(err_msg)
            sys.exit(1)
            
        print(f"\nFull Transcript:\n{full_transcript}\n", file=sys.stderr)

        final_timestamps_for_srt = []
        srt_generation_type = "none"

        if segment_timestamps_from_model:
            # ... (segment processing from v17d) ...
            srt_generation_type = "segment"
            print(f"Found {len(segment_timestamps_from_model)} segment timestamps. Using 'start' and 'end' keys (expected in seconds).", file=sys.stderr)
            for idx, ts_item in enumerate(segment_timestamps_from_model):
                text_val = ts_item.get('segment')
                start_s_seconds_raw = ts_item.get('start') 
                end_s_seconds_raw = ts_item.get('end')     
                if start_s_seconds_raw is not None and end_s_seconds_raw is not None and text_val is not None:
                    try:
                        final_timestamps_for_srt.append({
                            'text_from_model': str(text_val),
                            'start_seconds': float(start_s_seconds_raw),
                            'end_seconds': float(end_s_seconds_raw)
                        })
                    except ValueError as e:
                        print(f"Debug: Could not convert segment offsets to float: start='{start_s_seconds_raw}', end='{end_s_seconds_raw}'. Error: {e}", file=sys.stderr)
                        continue
                else:
                    print(f"Debug: Malformed segment timestamp item (missing 'start', 'end', or 'segment'): {ts_item}", file=sys.stderr)
        
        if not final_timestamps_for_srt and word_timestamps_from_model:
            # ... (word processing from v17d) ...
            pass

        if global_offset_seconds != 0.0 and final_timestamps_for_srt:
            # ... (offset application from v17d) ...
            print(f"Applying global start offset of {global_offset_seconds:.3f} seconds to all timestamps.", file=sys.stderr)
            for ts_data in final_timestamps_for_srt:
                ts_data['start_seconds'] += global_offset_seconds
                ts_data['end_seconds'] += global_offset_seconds
                if ts_data['start_seconds'] < 0: ts_data['start_seconds'] = 0.0
                if ts_data['end_seconds'] < 0: ts_data['end_seconds'] = 0.01 
                if ts_data['end_seconds'] <= ts_data['start_seconds']:
                    ts_data['end_seconds'] = ts_data['start_seconds'] + 0.1


        if final_timestamps_for_srt:
            if srt_generation_type == "segment":
                generate_srt_from_processed_timestamps(final_timestamps_for_srt, srt_path, timestamp_type="segment")
            elif srt_generation_type == "word":
                generate_srt_from_processed_timestamps(final_timestamps_for_srt, srt_path, timestamp_type="word")
        else:
            err_msg = "No segment or word timestamps processed"
            print(f"Error: {err_msg}. Cannot generate detailed SRT.", file=sys.stderr)
            if full_transcript is not None:
                # ... (fallback full transcript SRT generation with offset from v17d) ...
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write("1\n")
                    adjusted_start_time_full = global_offset_seconds
                    adjusted_end_time_full = audio_duration_seconds + global_offset_seconds
                    if adjusted_start_time_full < 0: adjusted_start_time_full = 0.0
                    if adjusted_end_time_full < adjusted_start_time_full: adjusted_end_time_full = adjusted_start_time_full + 0.1
                    f.write(f"{format_time_srt(adjusted_start_time_full)} --> {format_time_srt(adjusted_end_time_full)}\n")
                    f.write(full_transcript.strip() + "\n\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(f"SRT file generated with full transcript at '{srt_path}'. Offset applied.", file=sys.stderr)
            else:
                 write_error_srt("No transcript text available")

        print(f"SRT file processing completed for '{srt_path}'", file=sys.stderr)

    except torch.cuda.OutOfMemoryError as e_oom:
        err_msg = f"CUDA OutOfMemoryError: {e_oom}"
        print(err_msg, file=sys.stderr)
        print("The audio file might be too long for your GPU's VRAM even with long audio settings.", file=sys.stderr)
        write_error_srt("CUDA OutOfMemoryError")
        sys.exit(1) 
    except SystemExit: # Allow sys.exit to propagate
        raise
    except Exception as e:
        err_msg = f"An error occurred: {e}"
        print(err_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        write_error_srt(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        if asr_model is not None:
            # ... (cleanup from v17d) ...
            if long_audio_settings_applied:
                try:
                    print("Reverting long audio settings...", file=sys.stderr)
                    asr_model.change_attention_model(self_attention_model="rel_pos")
                    asr_model.change_subsampling_conv_chunking_factor(-1)
                    print("Long audio settings reverted.", file=sys.stderr)
                except Exception as revert_e:
                    print(f"Warning: Failed to revert long audio settings: {revert_e}", file=sys.stderr)
            try:
                if hasattr(asr_model, 'cpu'): asr_model.cpu()
                del asr_model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print("Model moved to CPU and CUDA cache cleared (if applicable).", file=sys.stderr)
            except Exception as cleanup_e:
                print(f"Error during model cleanup: {cleanup_e}", file=sys.stderr)

if __name__ == "__main__":
    main()

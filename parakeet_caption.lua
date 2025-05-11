-- parakeet_mpv_ffmpeg_venv.lua (v11f - Hotkeys Alt+4/5/6/7, Faster SRT load, shutdown cleanup)
-- Lua script for MPV to transcribe audio using parakeet_transcribe.py.
-- MODIFIED:
-- - Changed hotkeys to Alt+1, Alt+2, Alt+3, Alt+4.
-- - Removed auto_load_and_cleanup_delay_seconds.
-- - SRT loading now attempted immediately after Python script finishes.
-- - Temporary file cleanup moved to MPV shutdown event.
-- FIX: Explicitly set -ar 16000 in the FFmpeg filter pass.
-- TEST: Using only 'loudnorm' for FFmpeg audio filters.

local mp = require 'mp'
local utils = require 'mp.utils'

-- ########## Configuration ##########
local python_exe = "C:/venvs/nemo_mpv_py312/Scripts/python.exe" -- Your Python executable path
local parakeet_script_path = "C:/Parakeet_Caption/parakeet_transcribe.py" -- Path to your parakeet_transcribe.py
local ffmpeg_path = "ffmpeg" -- or full path to ffmpeg.exe
local ffprobe_path = "ffprobe" -- or full path to ffprobe.exe
local temp_dir = "C:/temp" -- Ensure this directory exists and is writable

-- Keybindings:
local key_binding_default = "Alt+4"                 -- Standard transcription
local key_binding_py_float32 = "Alt+5"            -- Python Float32 precision
local key_binding_ffmpeg_preprocess = "Alt+6"     -- FFmpeg Preprocessing
local key_binding_ffmpeg_py_float32 = "Alt+7"   -- FFmpeg Preprocessing + Python Float32

-- FFmpeg audio filter chain for pre-processing mode
local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"
-- ###################################

-- Table to store paths of temporary files for cleanup on shutdown
local files_to_cleanup_on_shutdown = {}

-- Helper function to safely convert a value to string for logging
local function to_str_safe(val)
    if val == nil then return "nil" end
    return tostring(val)
end

-- Ensure temp directory exists
if not utils.file_info(temp_dir) then
    mp.msg.warn("[parakeet_mpv] Temporary directory does not exist: " .. temp_dir)
    local mkdir_cmd
    if package.config:sub(1,1) == '\\' then -- Windows
        if temp_dir:match("^[A-Za-z]:$") then -- Check if it's a root drive like C:
             mp.msg.warn("[parakeet_mpv] Cannot 'mkdir' a root drive like '" .. temp_dir .. "'. Please ensure it's accessible and not a root drive itself for mkdir.")
        else
            -- For Windows, use cmd /C to handle spaces in paths correctly for mkdir
            mkdir_cmd = string.format('cmd /C "if not exist "%s" mkdir "%s""', temp_dir:gsub("/", "\\"), temp_dir:gsub("/", "\\"))
            local _, err_code = os.execute(mkdir_cmd)
            if err_code == 0 then
                 mp.msg.info("[parakeet_mpv] Attempted to create temp directory: " .. temp_dir)
            else
                 mp.msg.warn("[parakeet_mpv] Failed to create temp directory. Exit Code: " .. to_str_safe(err_code))
            end
        end
    else -- Linux/macOS
        mkdir_cmd = string.format('mkdir -p "%s"', temp_dir)
        os.execute(mkdir_cmd) -- os.execute on Linux/macOS returns true/false or nil+message+code
        mp.msg.info("[parakeet_mpv] Attempted to create temp directory: " .. temp_dir)
    end
end

-- Internal logging function
local function log(level, ...)
    local args = {...}
    local msg_parts = {}
    for i = 1, #args do
        table.insert(msg_parts, to_str_safe(args[i]))
    end
    local message = table.concat(msg_parts, " ")
    local prefixed_msg = "[parakeet_mpv] " .. message
    mp.msg[level](prefixed_msg)
    if level == "error" then mp.osd_message("Parakeet Error: " .. message, 7)
    elseif level == "warn" then mp.osd_message("Parakeet Warning: " .. message, 5)
    elseif level == "info" then mp.osd_message("Parakeet: " .. message, 3)
    end
end

-- Function to safely remove a file
local function safe_remove(filepath, description)
    if not filepath then return end
    if utils.file_info(filepath) then
        local success, err_msg = os.remove(filepath)
        if success then log("debug", "Cleaned up temporary file (" .. (description or "") .. "): ", filepath)
        else log("warn", "Failed to remove temporary file (" .. (description or "") .. "): ", filepath, " - Error: ", (err_msg or "unknown")) end
    else log("debug", "Temporary file (" .. (description or "") .. ") not found for removal, skipping: ", filepath) end
end

-- Function to get audio stream start offset and specific index
local function get_audio_stream_info(media_path, target_language_code)
    local ffprobe_cmd_args = { ffprobe_path, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a", media_path }
    log("debug", "Running ffprobe to find audio streams: ", table.concat(ffprobe_cmd_args, " "))
    local res = utils.subprocess({args = ffprobe_cmd_args, cancellable = false, capture_stdout = true, capture_stderr = true})
    if res.error or res.status ~= 0 or not res.stdout then
        log("warn", "ffprobe command failed. Error: ", to_str_safe(res.error), ", Status: ", to_str_safe(res.status))
        if res.stderr and string.len(res.stderr) > 0 then log("warn", "ffprobe stderr: ", res.stderr) end
        return 0.0, nil
    end
    local success, data = pcall(utils.parse_json, res.stdout)
    if not success or not data or not data.streams or #data.streams == 0 then
        log("warn", "Failed to parse ffprobe JSON or no audio streams. Data: ", to_str_safe(res.stdout))
        return 0.0, nil
    end
    local selected_stream_absolute_index, selected_stream_start_time = nil, 0.0
    if target_language_code then
        for _, stream in ipairs(data.streams) do
            if stream.codec_type == "audio" and stream.tags and stream.tags.language and stream.tags.language:lower():match(target_language_code:lower()) then
                selected_stream_absolute_index = tostring(stream.index)
                selected_stream_start_time = tonumber(stream.start_time) or 0.0
                log("info", "Found target lang '", target_language_code, "' (abs idx ", selected_stream_absolute_index, ", start_time ", selected_stream_start_time, ")")
                return selected_stream_start_time, selected_stream_absolute_index
            end
        end
        log("warn", "Target lang '", target_language_code, "' not found by ffprobe.")
    end
    log("info", "No specific lang match or no target. Using first audio stream.")
    for _, stream in ipairs(data.streams) do
        if stream.codec_type == "audio" then
            selected_stream_absolute_index = tostring(stream.index)
            selected_stream_start_time = tonumber(stream.start_time) or 0.0
            log("info", "Using first audio (abs idx ", selected_stream_absolute_index, ", start_time ", selected_stream_start_time, ")")
            return selected_stream_start_time, selected_stream_absolute_index
        end
    end
    log("warn", "No audio streams found by ffprobe at all.")
    return 0.0, nil
end

-- Core transcription logic
local function do_transcription_core(force_python_float32_flag, apply_ffmpeg_filters_flag)
    -- Validations
    if not utils.file_info(python_exe) then log("error", "Python executable not found: ", python_exe) return end
    if not utils.file_info(parakeet_script_path) then log("error", "Parakeet script not found: '", parakeet_script_path) return end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then log("error", "FFmpeg executable not found: ", ffmpeg_path) return end
    if ffprobe_path ~= "ffprobe" and not utils.file_info(ffprobe_path) then log("error", "FFprobe executable not found: ", ffprobe_path) return end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory.") return end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then log("error", "No file is currently playing.") return end

    local file_name_with_ext = current_media_path:match("([^/\\]+)$") or "unknown_file"
    local base_name = file_name_with_ext:match("(.+)%.[^%.]+$") or file_name_with_ext
    local media_dir = ""
    local path_sep_pos = current_media_path:match("^.*[/\\]()")
    if path_sep_pos then media_dir = current_media_path:sub(1, path_sep_pos -1)
    else local cwd = utils.getcwd() if cwd then media_dir = cwd end end
    if media_dir ~= "" and not media_dir:match("[/\\]$") then media_dir = media_dir .. package.config:sub(1,1) end -- Add trailing slash if missing

    local srt_output_path = utils.join_path(media_dir, base_name .. ".srt")
    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_") -- Sanitize for use in temp filenames

    local temp_audio_raw_path = utils.join_path(temp_dir, sanitized_base_name .. "_audio_raw.wav")
    local temp_audio_for_python = temp_audio_raw_path

    -- Add raw audio path to cleanup list (will be cleaned on shutdown)
    table.insert(files_to_cleanup_on_shutdown, temp_audio_raw_path)

    local mode_description = "Standard (Alt+1)"
    if force_python_float32_flag and apply_ffmpeg_filters_flag then mode_description = "FFmpeg Preproc + Python Float32 (Alt+4)"
    elseif force_python_float32_flag then mode_description = "Python Float32 (Alt+2)"
    elseif apply_ffmpeg_filters_flag then mode_description = "FFmpeg Preproc (Alt+3)"
    end

    mp.osd_message("Parakeet (" .. mode_description .. "): Analyzing audio streams...", 3)
    log("info", "Step 0: Analyzing audio streams with FFprobe for transcription (" .. mode_description .. ")...")

    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng") -- Try "eng" first
    log("info", "Determined audio stream start offset: ", audio_stream_offset_seconds, "s. Specific stream absolute index for FFmpeg: ", to_str_safe(specific_audio_absolute_idx))

    mp.osd_message("Parakeet: Preparing audio with FFmpeg...", 5)
    log("info", "Step 1: Extracting audio track with FFmpeg...")
    log("info", "Outputting raw temporary audio to: ", temp_audio_raw_path)

    local ffmpeg_common_args = {"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y"}
    local ffmpeg_args_extract
    local ffmpeg_map_value_for_log

    if specific_audio_absolute_idx then
        ffmpeg_map_value_for_log = "0:" .. specific_audio_absolute_idx
        log("info", "Using specific stream map for FFmpeg extraction: ", ffmpeg_map_value_for_log)
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    else
        log("warn", "No specific stream index from ffprobe. Attempting FFmpeg default English mapping first for extraction.")
        ffmpeg_map_value_for_log = "0:a:m:language:eng" -- Try to map English audio by metadata
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    end

    for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
    table.insert(ffmpeg_args_extract, temp_audio_raw_path)

    log("debug", "Running FFmpeg extraction with map '", ffmpeg_map_value_for_log, "': ", table.concat(ffmpeg_args_extract, " "))
    local ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true })

    -- Fallback if initial mapping (e.g., English or specific index) fails
    if (ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0) and not specific_audio_absolute_idx then
        log("warn", "FFmpeg (map '", ffmpeg_map_value_for_log ,"') extraction failed. Trying fallback map...")
        if ffmpeg_res_extract.stderr and string.len(ffmpeg_res_extract.stderr) > 0 then log("warn", "FFmpeg Stderr (attempt 1): ", ffmpeg_res_extract.stderr) end
        mp.osd_message("Parakeet: Specific/English audio map failed, trying fallback map...", 3)

        -- Re-run ffprobe for the generic first audio stream if the targeted one failed
        local fallback_offset_for_fb, fallback_idx_for_fb = get_audio_stream_info(current_media_path, nil) -- nil for any audio
        if fallback_offset_for_fb ~= audio_stream_offset_seconds then
            log("info", "Updating audio offset for fallback to: ", fallback_offset_for_fb, "s.")
            audio_stream_offset_seconds = fallback_offset_for_fb
        end
        
        if fallback_idx_for_fb then
            ffmpeg_map_value_for_log = "0:" .. fallback_idx_for_fb
            log("info", "Using specific index for fallback map: ", ffmpeg_map_value_for_log)
        else
            ffmpeg_map_value_for_log = "0:a:0?" -- Select first available audio stream, if it exists
            log("warn", "FFprobe found no audio streams for fallback index. Using generic 0:a:0? for FFmpeg.")
        end
        
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
        table.insert(ffmpeg_args_extract, temp_audio_raw_path)

        log("debug", "Running FFmpeg extraction (fallback map '", ffmpeg_map_value_for_log, "'): ", table.concat(ffmpeg_args_extract, " "))
        ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true }) 

        if ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then
            log("error", "FFmpeg fallback audio extraction ('", ffmpeg_map_value_for_log ,"') also failed. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
            mp.osd_message("Parakeet: Failed to extract audio with FFmpeg.", 7)
            safe_remove(temp_audio_raw_path, "Failed FFmpeg raw temp audio") -- Immediate cleanup on error
            return
        end
    elseif ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then 
        -- This case handles failure when a specific_audio_absolute_idx was initially found and used
        log("error", "FFmpeg extraction with specific map '", ffmpeg_map_value_for_log, "' failed. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        mp.osd_message("Parakeet: Failed to extract audio with FFmpeg (specific map).", 7)
        safe_remove(temp_audio_raw_path, "Failed FFmpeg raw temp audio") -- Immediate cleanup on error
        return
    end
    
    -- Check if the raw audio file was actually created and has content
    if not utils.file_info(temp_audio_raw_path) or utils.file_info(temp_audio_raw_path).size == 0 then
        log("error", "FFmpeg ran but raw temporary audio file '", temp_audio_raw_path, "' was not created or is empty.")
        mp.osd_message("Parakeet: FFmpeg failed to produce raw audio file.", 7)
        -- temp_audio_raw_path is already in files_to_cleanup_on_shutdown, so no extra safe_remove needed here
        return
    end
    log("info", "FFmpeg raw audio extraction successful: ", temp_audio_raw_path)

    -- Apply FFmpeg filters if the flag is set
    if apply_ffmpeg_filters_flag then
        temp_audio_for_python = utils.join_path(temp_dir, sanitized_base_name .. "_audio_filtered.wav")
        -- Add filtered audio path to cleanup list if it's different from raw (it always should be if this block runs)
        if temp_audio_raw_path ~= temp_audio_for_python then
             table.insert(files_to_cleanup_on_shutdown, temp_audio_for_python)
        end
        log("info", "Step 1.5: Applying FFmpeg audio filters: ", ffmpeg_audio_filters)
        mp.osd_message("Parakeet: Applying FFmpeg audio filters...", 5)
        local ffmpeg_args_filter = {
            ffmpeg_path,
            "-i", temp_audio_raw_path,
            "-af", ffmpeg_audio_filters,
            "-ar", "16000", -- Explicitly set output sample rate for filtered audio
            "-y", 
            temp_audio_for_python 
        }
        log("debug", "Running FFmpeg filter pass: ", table.concat(ffmpeg_args_filter, " "))
        local ffmpeg_res_filter = utils.subprocess({ args = ffmpeg_args_filter, cancellable = false, capture_stdout = true, capture_stderr = true })

        if ffmpeg_res_filter.error or ffmpeg_res_filter.status ~= 0 then
            log("error", "FFmpeg audio filtering failed. Error: ", to_str_safe(ffmpeg_res_filter.error), ", Status: ", to_str_safe(ffmpeg_res_filter.status))
            if ffmpeg_res_filter.stderr and string.len(ffmpeg_res_filter.stderr) > 0 then log("error", "FFmpeg Filter Stderr: ", ffmpeg_res_filter.stderr) end
            mp.osd_message("Parakeet: FFmpeg audio filtering failed.", 7)
            -- Clean up both potential temp files on error
            safe_remove(temp_audio_raw_path, "Temp raw audio after filter fail") -- Immediate cleanup
            if temp_audio_raw_path ~= temp_audio_for_python then
                safe_remove(temp_audio_for_python, "Failed FFmpeg filtered temp audio") -- Immediate cleanup
            end
            return
        end
        if not utils.file_info(temp_audio_for_python) or utils.file_info(temp_audio_for_python).size == 0 then
            log("error", "FFmpeg filtering ran but final audio file '", temp_audio_for_python, "' is missing or empty.")
            mp.osd_message("Parakeet: FFmpeg filtering produced no audio.", 7)
            -- Files are already in files_to_cleanup_on_shutdown, so they will be cleaned eventually.
            return
        end
        log("info", "FFmpeg audio filtering successful. Final audio for transcription: ", temp_audio_for_python)
    end

    mp.osd_message("Parakeet (" .. mode_description .. "): Transcribing... This may take a while.", 7)
    log("info", "Step 2: Starting Parakeet transcription for: ", file_name_with_ext, " (Using audio: ", temp_audio_for_python, ")")

    local python_command_args = {
        python_exe,
        parakeet_script_path,
        temp_audio_for_python, 
        srt_output_path,
        "--audio_start_offset", tostring(audio_stream_offset_seconds) 
    }
    if force_python_float32_flag then
        table.insert(python_command_args, "--force_float32")
    end

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
    -- This call blocks until the Python script finishes
    local python_res = utils.subprocess({ args = python_command_args, cancellable = false, capture_stdout = true, capture_stderr = true })

    if python_res.error then 
        log("error", "Failed to launch Parakeet Python script: ", (python_res.error or "Unknown error"))
        if python_res.stderr and string.len(python_res.stderr) > 0 then
             log("error", "Stderr from Python launch failure: ", python_res.stderr)
        end
        mp.osd_message("Parakeet: Failed to launch Python. Check console.", 7)
    else
        log("info", "Parakeet Python script finished (PID: ", (python_res.pid or "unknown"), "). Status: ", to_str_safe(python_res.status))
        if python_res.stdout and string.len(python_res.stdout) > 0 then log("debug", "Python script stdout: ", python_res.stdout) end
        if python_res.stderr and string.len(python_res.stderr) > 0 then log("debug", "Python script stderr: ", python_res.stderr) end

        if python_res.status ~= nil and python_res.status ~= 0 then
             log("warn", "Python script exited with an error. Status: ", to_str_safe(python_res.status))
             mp.osd_message("Parakeet: Python script error. Check console.", 7)
             -- SRT might still be an error SRT (generated by the python script), so we attempt to load it anyway.
        end

        log("info", "Attempting to load SRT immediately: ", srt_output_path)
        if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
            mp.commandv("sub-add", srt_output_path, "auto")
            mp.osd_message("Parakeet: Loaded " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 3)
        elseif utils.file_info(srt_output_path) then 
            log("warn", "SRT file found but is empty: ", srt_output_path)
            mp.osd_message("Parakeet: SRT file empty. Check console.", 5)
        else
            log("warn", "SRT file not found after Python script execution: ", srt_output_path)
            mp.osd_message("Parakeet: SRT not found. Check Python logs.", 7)
        end
    end
    -- Temporary audio files listed in files_to_cleanup_on_shutdown will be cleaned when MPV closes.
    log("info", "Transcription process complete. Temporary files will be cleaned on MPV shutdown.")
end

-- Wrapper functions for keybindings
local function transcribe_default_wrapper()
    do_transcription_core(false, false) -- python_float32=false, ffmpeg_filters=false
end

local function transcribe_py_float32_wrapper()
    do_transcription_core(true, false) -- python_float32=true, ffmpeg_filters=false
end

local function transcribe_ffmpeg_preprocess_wrapper()
    do_transcription_core(false, true) -- python_float32=false, ffmpeg_filters=true
end

local function transcribe_ffmpeg_py_float32_wrapper()
    do_transcription_core(true, true) -- python_float32=true, ffmpeg_filters=true
end

mp.add_key_binding(key_binding_default, "parakeet-transcribe-default", transcribe_default_wrapper)
mp.add_key_binding(key_binding_py_float32, "parakeet-transcribe-py-float32", transcribe_py_float32_wrapper)
mp.add_key_binding(key_binding_ffmpeg_preprocess, "parakeet-transcribe-ffmpeg-preprocess", transcribe_ffmpeg_preprocess_wrapper)
mp.add_key_binding(key_binding_ffmpeg_py_float32, "parakeet-transcribe-ffmpeg-py-float32", transcribe_ffmpeg_py_float32_wrapper)

log("info", "Parakeet (Multi-Mode, v11f - Hotkeys Alt+1/2/3/4) script loaded.")
log("info", "SRT will be loaded immediately after transcription.")
log("info", "Temporary files will be cleaned up on MPV shutdown.")
log("info", "Press '", key_binding_default, "' for Standard Transcription.")
log("info", "Press '", key_binding_py_float32, "' for Python Float32 Precision.")
log("info", "Press '", key_binding_ffmpeg_preprocess, "' for FFmpeg Preprocessing (Default Python Precision).")
log("info", "Press '", key_binding_ffmpeg_py_float32, "' for FFmpeg Preprocessing + Python Float32 Precision.")
log("info", "Using Python from: ", python_exe)
log("info", "Using FFmpeg from: ", ffmpeg_path)
log("info", "Using FFprobe from: ", ffprobe_path)
log("info", "Parakeet script: ", parakeet_script_path)
log("info", "Temporary file directory: ", temp_dir)
log("info", "FFmpeg pre-processing filters: ", ffmpeg_audio_filters)

mp.register_event("shutdown", function()
    log("info", "MPV shutdown event. Cleaning up temporary Parakeet files...")
    if #files_to_cleanup_on_shutdown > 0 then
        for i, filepath in ipairs(files_to_cleanup_on_shutdown) do
            safe_remove(filepath, "Shutdown cleanup")
        end
        -- Clear the table for the next session (though script reloads on mpv start anyway)
        files_to_cleanup_on_shutdown = {} 
    else
        log("info", "No temporary files registered for cleanup.")
    end
    log("info", "Parakeet shutdown cleanup finished.")
end)

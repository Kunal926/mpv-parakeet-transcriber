local mp = require 'mp'
local utils = require 'mp.utils'

-- ########## Configuration ##########
-- These paths should be configured by the user.

--- Path to the Python executable within the virtual environment.
-- @type string
local python_exe = "C:/venvs/nemo_mpv_py312/Scripts/python.exe"

--- Path to the parakeet_transcribe.py script.
-- @type string
local parakeet_script_path = "C:/Parakeet_Caption/parakeet_transcribe.py"

--- Path to the FFmpeg executable. Can be "ffmpeg" if in PATH.
-- @type string
local ffmpeg_path = "ffmpeg"

--- Path to the FFprobe executable. Can be "ffprobe" if in PATH.
-- @type string
local ffprobe_path = "ffprobe"

--- Directory for storing temporary audio files.
-- This directory must exist and be writable by MPV.
-- @type string
local temp_dir = "C:/temp"

-- Keybindings for different transcription modes.
local key_binding_default = "Alt+4"                 -- Standard transcription (no FFmpeg preprocessing, default Python precision)
local key_binding_py_float32 = "Alt+5"            -- Python Float32 precision (no FFmpeg preprocessing)
local key_binding_ffmpeg_preprocess = "Alt+6"     -- FFmpeg Preprocessing (default Python precision)
local key_binding_ffmpeg_py_float32 = "Alt+7"   -- FFmpeg Preprocessing + Python Float32 Precision

--- FFmpeg audio filter chain for pre-processing mode.
-- Example: "loudnorm=I=-16:LRA=7:TP=-1.5" for loudness normalization.
-- Other filters like denoisers (anlmdn, afftdn) or compressors can be added.
-- @type string
local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"
-- ###################################

--- Table to store paths of temporary files for cleanup on MPV shutdown.
-- @type table<string>
local files_to_cleanup_on_shutdown = {}

--- Safely converts a value to its string representation.
-- Handles nil values by returning "nil".
-- @param val any The value to convert.
-- @return string The string representation of the value.
local function to_str_safe(val)
    if val == nil then return "nil" end
    return tostring(val)
end

-- Ensure the temporary directory exists. Create it if it doesn't.
if not utils.file_info(temp_dir) then
    mp.msg.warn("[parakeet_mpv] Temporary directory does not exist: " .. temp_dir)
    local mkdir_cmd
    if package.config:sub(1,1) == '\\' then -- Windows OS
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
        os.execute(mkdir_cmd)
        mp.msg.info("[parakeet_mpv] Attempted to create temp directory: " .. temp_dir)
    end
end

--- Internal logging function for the script.
-- Prefixes messages with "[parakeet_mpv]" and sends them to MPV's console.
-- Also displays OSD messages for different log levels.
-- @param level string The log level (e.g., "info", "warn", "error", "debug").
-- @param ... any Variadic arguments forming the log message.
local function log(level, ...)
    local args = {...}
    local msg_parts = {}
    for i = 1, #args do
        table.insert(msg_parts, to_str_safe(args[i]))
    end
    local message = table.concat(msg_parts, " ")
    local prefixed_msg = "[parakeet_mpv] " .. message
    mp.msg[level](prefixed_msg) -- Log to MPV console

    -- Display On-Screen Display (OSD) messages for user feedback
    if level == "error" then mp.osd_message("Parakeet Error: " .. message, 7) -- Duration 7 seconds
    elseif level == "warn" then mp.osd_message("Parakeet Warning: " .. message, 5) -- Duration 5 seconds
    elseif level == "info" then mp.osd_message("Parakeet: " .. message, 3) -- Duration 3 seconds
    end
end

--- Safely removes a file from the filesystem.
-- Logs the action (success or failure).
-- @param filepath string The path to the file to be removed.
-- @param description string (optional) A description of the file for logging purposes.
local function safe_remove(filepath, description)
    if not filepath then return end
    if utils.file_info(filepath) then
        local success, err_msg = os.remove(filepath)
        if success then
            log("debug", "Cleaned up temporary file (" .. (description or "unspecified") .. "): ", filepath)
        else
            log("warn", "Failed to remove temporary file (" .. (description or "unspecified") .. "): ", filepath, " - Error: ", (err_msg or "unknown"))
        end
    else
        log("debug", "Temporary file (" .. (description or "unspecified") .. ") not found for removal, skipping: ", filepath)
    end
end

--- Retrieves audio stream information using ffprobe.
-- Specifically looks for the start time of the audio stream and its absolute index.
-- Prioritizes a target language if specified.
-- @param media_path string The full path to the media file.
-- @param target_language_code string (optional) The ISO 639-1/2 language code (e.g., "eng", "jpn") to prioritize.
-- @return number The start time of the selected audio stream in seconds (defaults to 0.0 if not found or error).
-- @return string|nil The absolute index of the selected audio stream as a string (e.g., "1", "2"), or nil if not found.
local function get_audio_stream_info(media_path, target_language_code)
    local ffprobe_cmd_args = {
        ffprobe_path,
        "-v", "quiet",              -- Suppress verbose output from ffprobe
        "-print_format", "json",    -- Output in JSON format
        "-show_streams",            -- Get information about all streams
        "-select_streams", "a",     -- Select only audio streams
        media_path
    }
    log("debug", "Running ffprobe to find audio streams: ", table.concat(ffprobe_cmd_args, " "))

    local res = utils.subprocess({args = ffprobe_cmd_args, cancellable = false, capture_stdout = true, capture_stderr = true})

    if res.error or res.status ~= 0 or not res.stdout then
        log("warn", "ffprobe command failed. Error: ", to_str_safe(res.error), ", Status: ", to_str_safe(res.status))
        if res.stderr and string.len(res.stderr) > 0 then log("warn", "ffprobe stderr: ", res.stderr) end
        return 0.0, nil -- Default to 0.0s offset and no specific index on error
    end

    local success, data = pcall(utils.parse_json, res.stdout) -- Safely parse JSON
    if not success or not data or not data.streams or #data.streams == 0 then
        log("warn", "Failed to parse ffprobe JSON or no audio streams found. Data: ", to_str_safe(res.stdout))
        return 0.0, nil
    end

    local selected_stream_absolute_index, selected_stream_start_time = nil, 0.0

    -- Try to find a stream matching the target language
    if target_language_code then
        for _, stream in ipairs(data.streams) do
            if stream.codec_type == "audio" and stream.tags and stream.tags.language and
               stream.tags.language:lower():match(target_language_code:lower()) then
                selected_stream_absolute_index = tostring(stream.index) -- ffprobe stream index
                selected_stream_start_time = tonumber(stream.start_time) or 0.0
                log("info", "Found target language audio stream '", target_language_code, "' (absolute index ", selected_stream_absolute_index, ", start_time ", selected_stream_start_time, "s)")
                return selected_stream_start_time, selected_stream_absolute_index
            end
        end
        log("warn", "Target language '", target_language_code, "' audio stream not found by ffprobe.")
    end

    -- If no target language or not found, use the first audio stream listed
    log("info", "No specific language match or no target language specified. Using the first available audio stream.")
    for _, stream in ipairs(data.streams) do
        if stream.codec_type == "audio" then
            selected_stream_absolute_index = tostring(stream.index)
            selected_stream_start_time = tonumber(stream.start_time) or 0.0
            log("info", "Using first audio stream found (absolute index ", selected_stream_absolute_index, ", start_time ", selected_stream_start_time, "s)")
            return selected_stream_start_time, selected_stream_absolute_index
        end
    end

    log("warn", "No audio streams found by ffprobe in the media file.")
    return 0.0, nil -- Should not happen if ffprobe found streams, but as a fallback
end

--- Core function to perform audio extraction, optional pre-processing, and transcription.
-- This function orchestrates the calls to ffprobe, ffmpeg, and the Python transcription script.
-- @param force_python_float32_flag boolean If true, passes "--force_float32" to the Python script.
-- @param apply_ffmpeg_filters_flag boolean If true, applies FFmpeg audio filters before transcription.
local function do_transcription_core(force_python_float32_flag, apply_ffmpeg_filters_flag)
    -- Step 0: Validations
    if not utils.file_info(python_exe) then log("error", "Python executable not found: ", python_exe) return end
    if not utils.file_info(parakeet_script_path) then log("error", "Parakeet Python script not found: '", parakeet_script_path, "'") return end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then log("error", "FFmpeg executable not found: ", ffmpeg_path) return end
    if ffprobe_path ~= "ffprobe" and not utils.file_info(ffprobe_path) then log("error", "FFprobe executable not found: ", ffprobe_path) return end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory.")
        return
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        log("error", "No media file is currently playing.")
        return
    end

    -- Derive file names and paths
    local file_name_with_ext = current_media_path:match("([^/\\]+)$") or "unknown_file"
    local base_name = file_name_with_ext:match("(.+)%.[^%.]+$") or file_name_with_ext -- Name without extension
    local media_dir = ""
    local path_sep_pos = current_media_path:match("^.*[/\\]()") -- Find last path separator
    if path_sep_pos then
        media_dir = current_media_path:sub(1, path_sep_pos -1) -- Directory of the media file
    else -- If no separator, assume current working directory (less common for mpv paths)
        local cwd = utils.getcwd()
        if cwd then media_dir = cwd end
    end
    if media_dir ~= "" and not media_dir:match("[/\\]$") then -- Ensure trailing slash for join_path
        media_dir = media_dir .. package.config:sub(1,1) -- OS-specific path separator
    end

    local srt_output_path = utils.join_path(media_dir, base_name .. ".srt") -- SRT next to media
    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_") -- Sanitize for temp file names

    local temp_audio_raw_path = utils.join_path(temp_dir, sanitized_base_name .. "_audio_raw.wav")
    local temp_audio_for_python = temp_audio_raw_path -- This will be the input to Python

    -- Ensure raw temp audio path is scheduled for cleanup, regardless of success/failure later
    table.insert(files_to_cleanup_on_shutdown, temp_audio_raw_path)

    local mode_description = "Standard (Alt+4)"
    if force_python_float32_flag and apply_ffmpeg_filters_flag then mode_description = "FFmpeg Preproc + Python Float32 (Alt+7)"
    elseif force_python_float32_flag then mode_description = "Python Float32 (Alt+5)"
    elseif apply_ffmpeg_filters_flag then mode_description = "FFmpeg Preproc (Alt+6)"
    end

    mp.osd_message("Parakeet (" .. mode_description .. "): Analyzing audio streams...", 3)
    log("info", "Step 0.1: Analyzing audio streams with FFprobe for transcription (" .. mode_description .. ")...")

    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng") -- Prioritize English
    log("info", "Determined audio stream start offset: ", audio_stream_offset_seconds, "s. Specific stream absolute index for FFmpeg: ", to_str_safe(specific_audio_absolute_idx))

    -- Step 1: Extract audio track with FFmpeg
    mp.osd_message("Parakeet: Preparing audio with FFmpeg...", 5)
    log("info", "Step 1.1: Extracting audio track with FFmpeg...")
    log("info", "Outputting raw temporary audio to: ", temp_audio_raw_path)

    -- Common FFmpeg args: no video, PCM 16-bit little-endian audio, 16kHz sample rate, 1 channel (mono), overwrite output
    local ffmpeg_common_args = {"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y"}
    local ffmpeg_args_extract
    local ffmpeg_map_value_for_log -- For logging which -map option was used

    if specific_audio_absolute_idx then
        ffmpeg_map_value_for_log = "0:" .. specific_audio_absolute_idx -- Map by absolute stream index from ffprobe
        log("info", "Using specific stream map for FFmpeg extraction: ", ffmpeg_map_value_for_log)
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    else
        log("warn", "No specific stream index from ffprobe (or target language not found). Attempting FFmpeg default English mapping first for extraction.")
        ffmpeg_map_value_for_log = "0:a:m:language:eng" -- Try to map by metadata: first audio stream with English language
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    end
    
    for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
    table.insert(ffmpeg_args_extract, temp_audio_raw_path) -- Output file
    
    log("debug", "Running FFmpeg extraction with map '", ffmpeg_map_value_for_log, "': ", table.concat(ffmpeg_args_extract, " "))
    local ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true })

    -- Fallback logic if the initial FFmpeg mapping failed (e.g., no 'eng' metadata or specific index was wrong)
    if (ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0) and not specific_audio_absolute_idx then
        log("warn", "FFmpeg (map '", ffmpeg_map_value_for_log ,"') extraction failed. Trying fallback map...")
        if ffmpeg_res_extract.stderr and string.len(ffmpeg_res_extract.stderr) > 0 then log("warn", "FFmpeg Stderr (attempt 1 with map '", ffmpeg_map_value_for_log, "'): ", ffmpeg_res_extract.stderr) end
        mp.osd_message("Parakeet: Specific/English audio map failed, trying fallback map...", 3)
        
        -- Re-run ffprobe for the generic first audio stream if the targeted one (e.g. 'eng') failed
        local fallback_offset_for_fb, fallback_idx_for_fb = get_audio_stream_info(current_media_path, nil) -- nil for any audio
        if fallback_offset_for_fb ~= audio_stream_offset_seconds then
            log("info", "Updating audio offset for fallback to: ", fallback_offset_for_fb, "s.")
            audio_stream_offset_seconds = fallback_offset_for_fb
        end
        
        if fallback_idx_for_fb then
            ffmpeg_map_value_for_log = "0:" .. fallback_idx_for_fb -- Use the absolute index of the first audio stream
            log("info", "Using specific index for fallback map: ", ffmpeg_map_value_for_log)
        else
            ffmpeg_map_value_for_log = "0:a:0?" -- Fallback to FFmpeg's first available audio stream (optional stream specifier)
            log("warn", "FFprobe found no audio streams for fallback index. Using generic '0:a:0?' for FFmpeg.")
        end
        
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
        table.insert(ffmpeg_args_extract, temp_audio_raw_path)

        log("debug", "Running FFmpeg extraction (fallback map '", ffmpeg_map_value_for_log, "'): ", table.concat(ffmpeg_args_extract, " "))
        ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true }) 

        if ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then
            log("error", "FFmpeg fallback audio extraction ('", ffmpeg_map_value_for_log ,"') also failed. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
            mp.osd_message("Parakeet: Failed to extract audio with FFmpeg even with fallback.", 7)
            -- safe_remove(temp_audio_raw_path, "Failed FFmpeg raw temp audio") -- Already in shutdown cleanup
            return
        end
    elseif ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then 
        -- This case handles failure when a specific_audio_absolute_idx was initially found and used, and it failed.
        log("error", "FFmpeg extraction with specific map '", ffmpeg_map_value_for_log, "' failed. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        mp.osd_message("Parakeet: Failed to extract audio with FFmpeg (specific map).", 7)
        -- safe_remove(temp_audio_raw_path, "Failed FFmpeg raw temp audio") -- Already in shutdown cleanup
        return
    end
    
    if not utils.file_info(temp_audio_raw_path) or utils.file_info(temp_audio_raw_path).size == 0 then
        log("error", "FFmpeg ran but raw temporary audio file '", temp_audio_raw_path, "' was not created or is empty.")
        mp.osd_message("Parakeet: FFmpeg failed to produce raw audio file.", 7)
        return
    end
    log("info", "FFmpeg raw audio extraction successful: ", temp_audio_raw_path)

    -- Step 1.5: Apply FFmpeg audio filters (optional)
    if apply_ffmpeg_filters_flag then
        temp_audio_for_python = utils.join_path(temp_dir, sanitized_base_name .. "_audio_filtered.wav")
        if temp_audio_raw_path ~= temp_audio_for_python then -- Should always be true if this block runs
             table.insert(files_to_cleanup_on_shutdown, temp_audio_for_python) -- Add filtered file for cleanup
        end
        log("info", "Step 1.5: Applying FFmpeg audio filters: ", ffmpeg_audio_filters)
        mp.osd_message("Parakeet: Applying FFmpeg audio filters...", 5)
        local ffmpeg_args_filter = {
            ffmpeg_path,
            "-i", temp_audio_raw_path,    -- Input is the raw extracted audio
            "-af", ffmpeg_audio_filters, -- Apply configured audio filters
            "-ar", "16000",               -- Explicitly set output sample rate for filtered audio
            "-y",                         -- Overwrite output file
            temp_audio_for_python         -- Output filtered audio
        }
        log("debug", "Running FFmpeg filter pass: ", table.concat(ffmpeg_args_filter, " "))
        local ffmpeg_res_filter = utils.subprocess({ args = ffmpeg_args_filter, cancellable = false, capture_stdout = true, capture_stderr = true })

        if ffmpeg_res_filter.error or ffmpeg_res_filter.status ~= 0 then
            log("error", "FFmpeg audio filtering failed. Error: ", to_str_safe(ffmpeg_res_filter.error), ", Status: ", to_str_safe(ffmpeg_res_filter.status))
            if ffmpeg_res_filter.stderr and string.len(ffmpeg_res_filter.stderr) > 0 then log("error", "FFmpeg Filter Stderr: ", ffmpeg_res_filter.stderr) end
            mp.osd_message("Parakeet: FFmpeg audio filtering failed.", 7)
            -- Raw and filtered temp files are already in files_to_cleanup_on_shutdown
            return
        end
        if not utils.file_info(temp_audio_for_python) or utils.file_info(temp_audio_for_python).size == 0 then
            log("error", "FFmpeg filtering ran but final audio file '", temp_audio_for_python, "' is missing or empty.")
            mp.osd_message("Parakeet: FFmpeg filtering produced no audio.", 7)
            return
        end
        log("info", "FFmpeg audio filtering successful. Final audio for transcription: ", temp_audio_for_python)
    end

    -- Step 2: Start Parakeet Python script for transcription
    mp.osd_message("Parakeet (" .. mode_description .. "): Transcribing... This may take a while.", 7)
    log("info", "Step 2.1: Starting Parakeet transcription for: ", file_name_with_ext, " (Using audio: ", temp_audio_for_python, ")")

    local python_command_args = {
        python_exe,
        parakeet_script_path,
        temp_audio_for_python, -- Input audio file (raw or filtered)
        srt_output_path,       -- Output SRT file path
        "--audio_start_offset", tostring(audio_stream_offset_seconds) -- Pass audio start offset
    }
    if force_python_float32_flag then
        table.insert(python_command_args, "--force_float32")
    end

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
    -- This is a blocking call: Lua script waits for Python to finish.
    local python_res = utils.subprocess({ args = python_command_args, cancellable = false, capture_stdout = true, capture_stderr = true })

    if python_res.error then 
        log("error", "Failed to launch Parakeet Python script: ", (python_res.error or "Unknown error"))
        if python_res.stderr and string.len(python_res.stderr) > 0 then
             log("error", "Stderr from Python launch failure: ", python_res.stderr)
        end
        mp.osd_message("Parakeet: Failed to launch Python. Check console.", 7)
    else
        log("info", "Parakeet Python script finished (PID: ", (python_res.pid or "unknown"), "). Status: ", to_str_safe(python_res.status))
        -- Log stdout/stderr from Python script for debugging if needed
        if python_res.stdout and string.len(python_res.stdout) > 0 then log("debug", "Python script stdout: ", python_res.stdout) end
        if python_res.stderr and string.len(python_res.stderr) > 0 then log("debug", "Python script stderr: ", python_res.stderr) end

        if python_res.status ~= nil and python_res.status ~= 0 then
             log("warn", "Python script exited with an error. Status: ", to_str_safe(python_res.status), ". Check Python script's own logging for details.")
             mp.osd_message("Parakeet: Python script error. Check console.", 7)
             -- Even if Python script reports an error, it might have produced an error SRT.
             -- So, we still attempt to load whatever SRT file might exist.
        end

        -- Step 3: Attempt to load the generated SRT file immediately
        log("info", "Attempting to load SRT immediately: ", srt_output_path)
        if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
            mp.commandv("sub-add", srt_output_path, "auto") -- Load subtitle, "auto" tries to guess encoding and select it
            mp.osd_message("Parakeet: Loaded " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 3)
        elseif utils.file_info(srt_output_path) then -- File exists but is empty
            log("warn", "SRT file found but is empty: ", srt_output_path, ". This might indicate a transcription problem or an error SRT with no content.")
            mp.osd_message("Parakeet: SRT file empty. Check console.", 5)
        else -- File not found
            log("warn", "SRT file not found after Python script execution: ", srt_output_path, ". Transcription may have failed.")
            mp.osd_message("Parakeet: SRT not found. Check Python logs.", 7)
        end
    end
    log("info", "Transcription process complete. Temporary audio files (if any) will be cleaned on MPV shutdown.")
end

--- Wrapper function for default transcription mode.
-- Calls `do_transcription_core` with standard settings.
local function transcribe_default_wrapper()
    do_transcription_core(false, false) -- force_python_float32_flag=false, apply_ffmpeg_filters_flag=false
end

--- Wrapper function for Python float32 precision transcription mode.
-- Calls `do_transcription_core` forcing float32 in Python.
local function transcribe_py_float32_wrapper()
    do_transcription_core(true, false) -- force_python_float32_flag=true, apply_ffmpeg_filters_flag=false
end

--- Wrapper function for FFmpeg pre-processing transcription mode.
-- Calls `do_transcription_core` with FFmpeg filters enabled.
local function transcribe_ffmpeg_preprocess_wrapper()
    do_transcription_core(false, true) -- force_python_float32_flag=false, apply_ffmpeg_filters_flag=true
end

--- Wrapper function for FFmpeg pre-processing and Python float32 precision mode.
-- Calls `do_transcription_core` with both FFmpeg filters and Python float32 enabled.
local function transcribe_ffmpeg_py_float32_wrapper()
    do_transcription_core(true, true) -- force_python_float32_flag=true, apply_ffmpeg_filters_flag=true
end

-- Register key bindings with MPV
mp.add_key_binding(key_binding_default, "parakeet-transcribe-default", transcribe_default_wrapper)
mp.add_key_binding(key_binding_py_float32, "parakeet-transcribe-py-float32", transcribe_py_float32_wrapper)
mp.add_key_binding(key_binding_ffmpeg_preprocess, "parakeet-transcribe-ffmpeg-preprocess", transcribe_ffmpeg_preprocess_wrapper)
mp.add_key_binding(key_binding_ffmpeg_py_float32, "parakeet-transcribe-ffmpeg-py-float32", transcribe_ffmpeg_py_float32_wrapper)

-- Log script loading and configuration information
log("info", "Parakeet (Multi-Mode, v11h - Hotkeys Alt+4/5/6/7) script loaded.")
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

--- Registers a function to be called when MPV is shutting down.
-- This is used to clean up any temporary files created by the script.
mp.register_event("shutdown", function()
    log("info", "MPV shutdown event. Cleaning up temporary Parakeet files...")
    if #files_to_cleanup_on_shutdown > 0 then
        for _, filepath in ipairs(files_to_cleanup_on_shutdown) do
            safe_remove(filepath, "Shutdown cleanup")
        end
        -- Clear the table for the next session (though script reloads on mpv start anyway)
        files_to_cleanup_on_shutdown = {} 
    else
        log("info", "No temporary files registered for cleanup.")
    end
    log("info", "Parakeet shutdown cleanup finished.")
end)

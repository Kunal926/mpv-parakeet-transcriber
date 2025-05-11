-- parakeet_mpv_ffmpeg_venv.lua (v14 - Further Reduced OSD)
-- Lua script for MPV to transcribe audio using parakeet_transcribe.py.
-- MODIFIED:
-- - Removed "FFmpeg: Extracting audio..." OSD message.
-- - Single default OSD duration (2 seconds).
-- - Removed OSDs for script load and keybindings.
-- - Changed srt_output_path to use temp_dir and sanitized_base_name

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
local key_binding_default = "Alt+4"               -- Standard transcription
local key_binding_py_float32 = "Alt+5"            -- Python Float32 precision
local key_binding_ffmpeg_preprocess = "Alt+6"     -- FFmpeg Preprocessing
local key_binding_ffmpeg_py_float32 = "Alt+7"   -- FFmpeg Preprocessing + Python Float32

--- FFmpeg audio filter chain for pre-processing mode.
-- @type string
local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"

--- Default OSD Message Duration (in seconds)
local osd_duration_default = 2
-- ###################################

--- Table to store paths of temporary files for cleanup on MPV shutdown.
-- @type table<string>
local files_to_cleanup_on_shutdown = {}

--- Safely converts a value to its string representation.
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
        if temp_dir:match("^[A-Za-z]:$") then
            mp.msg.warn("[parakeet_mpv] Cannot 'mkdir' a root drive like '" .. temp_dir .. "'. Please ensure it's accessible.")
        else
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
end

--- Safely removes a file from the filesystem.
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
-- @param media_path string The full path to the media file.
-- @param target_language_code string (optional) The ISO 639-1/2 language code.
-- @return number The start time of the selected audio stream in seconds.
-- @return string|nil The absolute index of the selected audio stream.
local function get_audio_stream_info(media_path, target_language_code)
    local ffprobe_cmd_args = {
        ffprobe_path, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "a", media_path
    }
    log("debug", "Running ffprobe: ", table.concat(ffprobe_cmd_args, " "))
    local res = utils.subprocess({args = ffprobe_cmd_args, cancellable = false, capture_stdout = true, capture_stderr = true})

    if res.error or res.status ~= 0 or not res.stdout then
        log("warn", "ffprobe command failed. Error: ", to_str_safe(res.error), ", Status: ", to_str_safe(res.status), ". Stderr: ", to_str_safe(res.stderr))
        return 0.0, nil
    end

    local success, data = pcall(utils.parse_json, res.stdout)
    if not success or not data or not data.streams or #data.streams == 0 then
        log("warn", "Failed to parse ffprobe JSON or no audio streams. Data: ", to_str_safe(res.stdout))
        return 0.0, nil
    end

    if target_language_code then
        for _, stream in ipairs(data.streams) do
            if stream.codec_type == "audio" and stream.tags and stream.tags.language and
               stream.tags.language:lower():match(target_language_code:lower()) then
                log("info", "Found target language '", target_language_code, "' (abs idx ", tostring(stream.index), ", start ", (tonumber(stream.start_time) or 0.0), "s)")
                return tonumber(stream.start_time) or 0.0, tostring(stream.index)
            end
        end
        log("warn", "Target language '", target_language_code, "' stream not found by ffprobe.")
    end

    for _, stream in ipairs(data.streams) do -- Fallback to first audio stream
        if stream.codec_type == "audio" then
            log("info", "Using first audio stream (abs idx ", tostring(stream.index), ", start ", (tonumber(stream.start_time) or 0.0), "s)")
            return tonumber(stream.start_time) or 0.0, tostring(stream.index)
        end
    end
    log("warn", "No audio streams found by ffprobe.")
    return 0.0, nil
end

--- Core function to perform audio extraction, optional pre-processing, and transcription.
-- @param force_python_float32_flag boolean If true, passes "--force_float32" to the Python script.
-- @param apply_ffmpeg_filters_flag boolean If true, applies FFmpeg audio filters.
local function do_transcription_core(force_python_float32_flag, apply_ffmpeg_filters_flag)
    -- Step 0: Display Mode OSD
    local mode_osd_message = "Parakeet: Unknown Mode"
    if not force_python_float32_flag and not apply_ffmpeg_filters_flag then
        mode_osd_message = "Parakeet: Standard Mode"
    elseif force_python_float32_flag and not apply_ffmpeg_filters_flag then
        mode_osd_message = "Parakeet: Float32 Mode"
    elseif not force_python_float32_flag and apply_ffmpeg_filters_flag then
        mode_osd_message = "Parakeet: FFmpeg Pre-processing Mode"
    elseif force_python_float32_flag and apply_ffmpeg_filters_flag then
        mode_osd_message = "Parakeet: FFmpeg Pre-processing + Float32 Mode"
    end
    mp.osd_message(mode_osd_message, osd_duration_default)

    -- Validations
    if not utils.file_info(python_exe) then log("error", "Python executable not found: ", python_exe) mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default) return end
    if not utils.file_info(parakeet_script_path) then log("error", "Parakeet Python script not found: '", parakeet_script_path, "'") mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default) return end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then log("error", "FFmpeg executable not found: ", ffmpeg_path) mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default) return end
    if ffprobe_path ~= "ffprobe" and not utils.file_info(ffprobe_path) then log("error", "FFprobe executable not found: ", ffprobe_path) mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default) return end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory.")
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        log("error", "No media file is currently playing.")
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end

    local file_name_with_ext = current_media_path:match("([^/\\]+)$") or "unknown_file"
    local base_name = file_name_with_ext:match("(.+)%.[^%.]+$") or file_name_with_ext
    -- local media_dir = "" -- Not needed if srt_output_path uses temp_dir
    -- local path_sep_pos = current_media_path:match("^.*[/\\]()")
    -- if path_sep_pos then media_dir = current_media_path:sub(1, path_sep_pos -1)
    -- else local cwd = utils.getcwd() if cwd then media_dir = cwd end end
    -- if media_dir ~= "" and not media_dir:match("[/\\]$") then media_dir = media_dir .. package.config:sub(1,1) end

    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_")
    -- MODIFIED LINE: srt_output_path now uses temp_dir and sanitized_base_name
    local srt_output_path = utils.join_path(temp_dir, sanitized_base_name .. ".srt")
    local temp_audio_raw_path = utils.join_path(temp_dir, sanitized_base_name .. "_audio_raw.wav")
    local temp_audio_for_python = temp_audio_raw_path
    table.insert(files_to_cleanup_on_shutdown, temp_audio_raw_path)
    -- Also add the SRT file to cleanup if it's in the temp directory
    table.insert(files_to_cleanup_on_shutdown, srt_output_path)


    log("info", "Analyzing audio streams with FFprobe...")
    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng")
    log("info", "Audio stream offset: ", audio_stream_offset_seconds, "s. Specific stream idx: ", to_str_safe(specific_audio_absolute_idx))

    -- mp.osd_message("FFmpeg: Extracting audio...", osd_duration_default) -- Removed as per user request
    log("info", "Extracting audio with FFmpeg to: ", temp_audio_raw_path)
    local ffmpeg_common_args = {"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y"}
    local ffmpeg_args_extract, ffmpeg_map_value_for_log

    if specific_audio_absolute_idx then
        ffmpeg_map_value_for_log = "0:" .. specific_audio_absolute_idx
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    else
        ffmpeg_map_value_for_log = "0:a:m:language:eng" -- Try to map by metadata: first audio stream with English
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    end
    for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
    table.insert(ffmpeg_args_extract, temp_audio_raw_path)
    log("debug", "Running FFmpeg extract (map '", ffmpeg_map_value_for_log, "'): ", table.concat(ffmpeg_args_extract, " "))
    local ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true })

    -- Fallback logic if the initial FFmpeg mapping failed
    if (ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0) and not specific_audio_absolute_idx then
        log("warn", "FFmpeg (map '", ffmpeg_map_value_for_log ,"') failed. Trying fallback. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        local fallback_offset, fallback_idx = get_audio_stream_info(current_media_path, nil) -- nil for any audio
        if fallback_offset ~= audio_stream_offset_seconds then
             log("info", "Updating audio offset for fallback to: ", fallback_offset, "s.")
             audio_stream_offset_seconds = fallback_offset
        end
        if fallback_idx then ffmpeg_map_value_for_log = "0:" .. fallback_idx
        else ffmpeg_map_value_for_log = "0:a:0?" log("warn", "FFprobe found no audio for fallback index. Using '0:a:0?'.") end
        
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
        table.insert(ffmpeg_args_extract, temp_audio_raw_path)
        log("debug", "Running FFmpeg extract (fallback map '", ffmpeg_map_value_for_log, "'): ", table.concat(ffmpeg_args_extract, " "))
        ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true })
    end

    if ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then
        log("error", "FFmpeg audio extraction failed. Map '", ffmpeg_map_value_for_log, "'. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end
    if not utils.file_info(temp_audio_raw_path) or utils.file_info(temp_audio_raw_path).size == 0 then
        log("error", "FFmpeg ran but raw temp audio '", temp_audio_raw_path, "' not created or empty.")
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end
    log("info", "FFmpeg raw audio extraction successful: ", temp_audio_raw_path)

    if apply_ffmpeg_filters_flag then
        temp_audio_for_python = utils.join_path(temp_dir, sanitized_base_name .. "_audio_filtered.wav")
        if temp_audio_raw_path ~= temp_audio_for_python then table.insert(files_to_cleanup_on_shutdown, temp_audio_for_python) end
        
        mp.osd_message("FFmpeg: Applying audio effects...", osd_duration_default)
        log("info", "Applying FFmpeg audio filters: ", ffmpeg_audio_filters, " to ", temp_audio_for_python)
        local ffmpeg_args_filter = {
            ffmpeg_path, "-i", temp_audio_raw_path, "-af", ffmpeg_audio_filters,
            "-ar", "16000", "-y", temp_audio_for_python
        }
        log("debug", "Running FFmpeg filter pass: ", table.concat(ffmpeg_args_filter, " "))
        local ffmpeg_res_filter = utils.subprocess({ args = ffmpeg_args_filter, cancellable = false, capture_stdout = true, capture_stderr = true })

        if ffmpeg_res_filter.error or ffmpeg_res_filter.status ~= 0 then
            log("error", "FFmpeg audio filtering failed. Stderr: ", to_str_safe(ffmpeg_res_filter.stderr))
            mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
            return
        end
        if not utils.file_info(temp_audio_for_python) or utils.file_info(temp_audio_for_python).size == 0 then
            log("error", "FFmpeg filtering ran but final audio '", temp_audio_for_python, "' missing or empty.")
            mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
            return
        end
        log("info", "FFmpeg audio filtering successful: ", temp_audio_for_python)
    end

    mp.osd_message("Parakeet: Transcribing audio...", osd_duration_default)
    log("info", "Starting Parakeet transcription for: ", file_name_with_ext, " (Audio: ", temp_audio_for_python, ", SRT to: ", srt_output_path, ")")
    local python_command_args = {
        python_exe, parakeet_script_path, temp_audio_for_python, srt_output_path,
        "--audio_start_offset", tostring(audio_stream_offset_seconds)
    }
    if force_python_float32_flag then table.insert(python_command_args, "--force_float32") end

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
    local python_res = utils.subprocess({ args = python_command_args, cancellable = false, capture_stdout = true, capture_stderr = true })

    if python_res.error or (python_res.status ~= nil and python_res.status ~= 0) then
        log("error", "Parakeet Python script execution failed. Error: ", to_str_safe(python_res.error), ", Status: ", to_str_safe(python_res.status), ". Stdout: ", to_str_safe(python_res.stdout), ". Stderr: ", to_str_safe(python_res.stderr))
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end
    
    log("info", "Python script finished. Status: ", to_str_safe(python_res.status), ". Stdout: ", to_str_safe(python_res.stdout), ". Stderr: ", to_str_safe(python_res.stderr))

    if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
        mp.commandv("sub-add", srt_output_path, "select")
        mp.osd_message("Parakeet: Subtitles loaded.", osd_duration_default)
        log("info", "SRT loaded: ", srt_output_path)
    else
        if not utils.file_info(srt_output_path) then
            log("error", "SRT file not found after Python script: ", srt_output_path)
        else
            log("error", "SRT file found but is empty: ", srt_output_path)
        end
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return -- Ensure we return here so "Transcription process complete" isn't logged on SRT failure
    end
    log("info", "Transcription process complete.")
end

local function transcribe_default_wrapper() do_transcription_core(false, false) end
local function transcribe_py_float32_wrapper() do_transcription_core(true, false) end
local function transcribe_ffmpeg_preprocess_wrapper() do_transcription_core(false, true) end
local function transcribe_ffmpeg_py_float32_wrapper() do_transcription_core(true, true) end

mp.add_key_binding(key_binding_default, "parakeet-transcribe-default", transcribe_default_wrapper)
mp.add_key_binding(key_binding_py_float32, "parakeet-transcribe-py-float32", transcribe_py_float32_wrapper)
mp.add_key_binding(key_binding_ffmpeg_preprocess, "parakeet-transcribe-ffmpeg-preprocess", transcribe_ffmpeg_preprocess_wrapper)
mp.add_key_binding(key_binding_ffmpeg_py_float32, "parakeet-transcribe-ffmpeg-py-float32", transcribe_ffmpeg_py_float32_wrapper)

log("info", "Parakeet (v14 - Further Reduced OSD - SRT Temp Fix) script loaded.")
log("info", "Keybindings: Default (" .. key_binding_default .. "), Float32 (" .. key_binding_py_float32 .. "), FFmpegPre (" .. key_binding_ffmpeg_preprocess .. "), FFmpegPre+Float32 (" .. key_binding_ffmpeg_py_float32 .. ")")
log("info", "Using Python: ", python_exe)
log("info", "Parakeet script: ", parakeet_script_path)
log("info", "Temp directory: ", temp_dir, " (SRT files will also be saved here)")
log("info", "FFmpeg filters: ", ffmpeg_audio_filters)
-- No OSD messages on script load as per user request.

mp.register_event("shutdown", function()
    log("info", "MPV shutdown: Cleaning up temporary Parakeet files...")
    if #files_to_cleanup_on_shutdown > 0 then
        for _, filepath in ipairs(files_to_cleanup_on_shutdown) do
            safe_remove(filepath, "Shutdown cleanup")
        end
        files_to_cleanup_on_shutdown = {} -- Clear the table after attempting removal
    else
        log("info", "No temporary files registered for cleanup.")
    end
    log("info", "Parakeet shutdown cleanup finished.")
end)

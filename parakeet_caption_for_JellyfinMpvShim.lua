-- Lua script for MPV to transcribe audio using parakeet_transcribe.py.
-- MODIFIED:
-- - Removed "FFmpeg: Extracting audio..." OSD message.
-- - Single default OSD duration.
-- - Removed OSDs for script load and keybindings.
-- - Changed srt_output_path to use temp_dir and sanitized_base_name

local mp = require 'mp'
local utils = require 'mp.utils'

-- ########## Configuration ##########
-- These paths should be configured by the user.

--- Path to the Python executable within the virtual environment.
-- This should be the full path to the `python.exe` (Windows) or `python` (Linux/macOS)
-- located inside the `Scripts` or `bin` directory of your Python virtual environment
-- where Riva Canary ASR / Parakeet is installed.
-- @type string
-- @example "C:/venvs/nemo_mpv_py312/Scripts/python.exe"
-- @example "/home/user/venvs/nemo_mpv_py312/bin/python"
local python_exe = "C:/venvs/nemo_mpv_py312/Scripts/python.exe"

--- Path to the parakeet_transcribe.py script.
-- This is the full path to the Python script responsible for performing the audio transcription.
-- @type string
-- @example "C:/Parakeet_Caption/parakeet_transcribe.py"
-- @example "/home/user/Parakeet_Caption/parakeet_transcribe.py"
local parakeet_script_path = "C:/Parakeet_Caption/parakeet_transcribe.py"

--- Path to the FFmpeg executable.
-- Can be set to just "ffmpeg" if the directory containing ffmpeg.exe (Windows) or ffmpeg (Linux/macOS)
-- is included in the system's PATH environment variable. Otherwise, provide the full path.
-- FFmpeg is used for extracting and pre-processing audio from the media file.
-- @type string
-- @example "ffmpeg"
-- @example "C:/ffmpeg/bin/ffmpeg.exe"
local ffmpeg_path = "ffmpeg"

--- Path to the FFprobe executable.
-- Can be set to just "ffprobe" if the directory containing ffprobe.exe (Windows) or ffprobe (Linux/macOS)
-- is included in the system's PATH environment variable. Otherwise, provide the full path.
-- FFprobe is used to gather information about audio streams in the media file.
-- @type string
-- @example "ffprobe"
-- @example "C:/ffmpeg/bin/ffprobe.exe"
local ffprobe_path = "ffprobe"

--- Directory for storing temporary audio and SRT files.
-- This directory must exist and be writable by MPV and the user running MPV.
-- Temporary WAV files extracted from the media and the generated SRT subtitle files
-- will be stored here. These files are cleaned up when MPV shuts down.
-- @type string
-- @example "C:/temp_mpv_transcripts"
-- @example "/tmp/mpv_parakeet_files"
local temp_dir = "C:/temp"

-- Keybindings for different transcription modes.
local key_binding_default = "Alt+4"             -- Standard transcription
local key_binding_py_float32 = "Alt+5"          -- Python Float32 precision
local key_binding_ffmpeg_preprocess = "Alt+6"   -- FFmpeg Preprocessing
local key_binding_ffmpeg_py_float32 = "Alt+7" -- FFmpeg Preprocessing + Python Float32

--- FFmpeg audio filter chain for pre-processing mode.
-- This string defines the audio filters FFmpeg will apply when the
-- FFmpeg pre-processing mode is activated (e.g., via Alt+6 or Alt+7).
-- Filters can help improve transcription accuracy by normalizing volume, reducing noise, etc.
-- The filter chain should be a valid FFmpeg -af argument.
-- @type string
-- @example "loudnorm=I=-16:LRA=7:TP=-1.5" (Loudness normalization)
-- @example "anlmdn" (Audio Non-Local Means de-noiser)
local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"

--- Default OSD Message Duration (in seconds).
-- All OSD messages displayed by this script will use this duration.
-- @type number
local osd_duration_default = 2
-- ###################################

--- Table to store paths of temporary files for cleanup on MPV shutdown.
-- When temporary audio or SRT files are created in `temp_dir`, their paths are added
-- to this table. The `mp.register_event("shutdown", ...)` function iterates
-- through this table to remove these files when MPV closes.
-- @type table<string>
local files_to_cleanup_on_shutdown = {}

--- Safely converts a value to its string representation.
-- Handles `nil` values by returning the string "nil", preventing errors
-- that would occur if `tostring(nil)` was called directly in concatenations.
-- @param val any The value to convert to a string.
-- @return string The string representation of the value, or "nil" if `val` is `nil`.
local function to_str_safe(val)
    if val == nil then return "nil" end
    return tostring(val)
end

-- Ensure the temporary directory exists. Attempt to create it if it doesn't.
-- This block checks for the existence of `temp_dir`. If not found, it tries
-- to create it using OS-specific commands. Warnings are logged if creation fails
-- or if `temp_dir` points to a root drive on Windows (which `mkdir` cannot create).
if not utils.file_info(temp_dir) then
    mp.msg.warn("[parakeet_mpv] Temporary directory does not exist: " .. temp_dir)
    local mkdir_cmd
    if package.config:sub(1,1) == '\\' then -- Windows OS
        if temp_dir:match("^[A-Za-z]:$") then -- Check if it's a root drive like C:
            mp.msg.warn("[parakeet_mpv] Cannot 'mkdir' a root drive like '" .. temp_dir .. "'. Please ensure it's accessible.")
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
-- Prefixes messages with "[parakeet_mpv]" and sends them to MPV's console
-- using the appropriate `mp.msg` level. This function no longer directly
-- handles OSD messages; OSD messages are now explicitly called where needed.
-- @param level string The log level (e.g., "info", "warn", "error", "debug").
-- @param ... any Variadic arguments forming the log message. These are converted to strings and concatenated.
-- @usage log("info", "Script initialized successfully.")
-- @usage log("error", "Failed to process file:", file_path, "Reason:", err_msg)
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
-- Checks if the file exists before attempting removal. Logs the action (success or failure)
-- using the `log` function at "debug" or "warn" level.
-- @param filepath string The path to the file to be removed. If `nil`, the function returns immediately.
-- @param description string (optional) A description of the file for logging purposes (e.g., "raw audio", "SRT file"). Defaults to "unspecified".
-- @usage safe_remove("/path/to/temp.wav", "temporary audio")
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
-- This function executes `ffprobe` to analyze a media file. It aims to find the
-- start time and the absolute index of a suitable audio stream.
-- It prioritizes an audio stream matching `target_language_code` if provided.
-- If no `target_language_code` is given or if a matching stream isn't found,
-- it defaults to the first audio stream listed by `ffprobe`.
-- @param media_path string The full path to the media file to be analyzed.
-- @param target_language_code string (optional) The ISO 639-1 (2-letter) or 639-2 (3-letter)
-- language code (e.g., "en", "eng", "ja", "jpn") to prioritize. Case-insensitive.
-- If `nil`, the first audio stream found will be used.
-- @return number The start time of the selected audio stream in seconds. Defaults to `0.0` if
-- not found, if an error occurs during `ffprobe` execution, or if JSON parsing fails.
-- @return string|nil The absolute index of the selected audio stream as a string (e.g., "0", "1", "2").
-- This index is suitable for use with FFmpeg's `-map 0:index` option. Returns `nil` if no suitable
-- stream is found or in case of an error.
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
               stream.tags.language:lower():match("^"..target_language_code:lower()) then -- Match prefix for codes like "eng" vs "en"
                log("info", "Found target language '", target_language_code, "' (abs idx ", tostring(stream.index), ", start ", (tonumber(stream.start_time) or 0.0), "s)")
                return tonumber(stream.start_time) or 0.0, tostring(stream.index)
            end
        end
        log("warn", "Target language '", target_language_code, "' stream not found by ffprobe.")
    end

    -- Fallback to first audio stream if target language not found or not specified
    for _, stream in ipairs(data.streams) do
        if stream.codec_type == "audio" then
            log("info", "Using first audio stream (abs idx ", tostring(stream.index), ", start ", (tonumber(stream.start_time) or 0.0), "s)")
            return tonumber(stream.start_time) or 0.0, tostring(stream.index)
        end
    end
    log("warn", "No audio streams found by ffprobe.")
    return 0.0, nil
end

--- Core function to perform audio extraction, optional pre-processing, and transcription.
-- This function orchestrates the entire transcription process:
-- 1. Displays an OSD message indicating the current transcription mode.
-- 2. Validates necessary executables and directories. If validation fails, logs an error and shows an OSD failure message.
-- 3. Retrieves the current media path and derives temporary file names. SRT files are now also saved in `temp_dir`.
-- 4. Uses `get_audio_stream_info` to determine the audio stream and its start offset, prioritizing English.
-- 5. Extracts the selected audio track to a temporary WAV file in `temp_dir` using FFmpeg. Includes fallback logic.
-- 6. If `apply_ffmpeg_filters_flag` is true, applies `ffmpeg_audio_filters` to the extracted WAV, creating another temporary WAV in `temp_dir`. An OSD message indicates this step.
-- 7. Invokes `parakeet_transcribe.py` with the appropriate temporary audio file, parameters (start offset, float32 flag), and the temporary SRT output path. An OSD message indicates transcription start.
-- 8. Attempts to load the generated SRT subtitle file from `temp_dir` into MPV using `sub-add` with `select`. OSD messages indicate success or failure.
-- 9. Logs progress and errors to the console.
-- Temporary audio and SRT files created in `temp_dir` are added to `files_to_cleanup_on_shutdown`.
--
-- @param force_python_float32_flag boolean If `true`, the "--force_float32" argument is passed to `parakeet_transcribe.py`.
-- @param apply_ffmpeg_filters_flag boolean If `true`, audio is pre-processed using `ffmpeg_audio_filters`.
-- @effects Creates temporary audio and SRT files in `temp_dir`.
-- @effects Loads the generated SRT file into MPV.
-- @effects Displays OSD messages to the user indicating progress and status.
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
    local function validation_failed(msg)
        log("error", msg)
        mp.osd_message("Parakeet: Transcription failed. Check MPV logs.", osd_duration_default)
        return
    end

    if not utils.file_info(python_exe) then return validation_failed("Python executable not found: " .. python_exe) end
    if not utils.file_info(parakeet_script_path) then return validation_failed("Parakeet Python script not found: '" .. parakeet_script_path .. "'") end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then return validation_failed("FFmpeg executable not found: " .. ffmpeg_path) end
    if ffprobe_path ~= "ffprobe" and not utils.file_info(ffprobe_path) then return validation_failed("FFprobe executable not found: " .. ffprobe_path) end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        return validation_failed("Temporary directory '" .. temp_dir .. "' does not exist or is not a directory.")
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        return validation_failed("No media file is currently playing.")
    end

    local file_name_with_ext = current_media_path:match("([^/\\]+)$") or "unknown_file"
    local base_name = file_name_with_ext:match("(.+)%.[^%.]+$") or file_name_with_ext
    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_")

    -- SRT output path is now in temp_dir
    local srt_output_path = utils.join_path(temp_dir, sanitized_base_name .. ".srt")
    local temp_audio_raw_path = utils.join_path(temp_dir, sanitized_base_name .. "_audio_raw.wav")
    local temp_audio_for_python = temp_audio_raw_path

    table.insert(files_to_cleanup_on_shutdown, temp_audio_raw_path)
    table.insert(files_to_cleanup_on_shutdown, srt_output_path) -- SRT is also temporary now

    log("info", "Analyzing audio streams with FFprobe...")
    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng") -- Prioritize English
    log("info", "Audio stream offset: ", audio_stream_offset_seconds, "s. Specific stream idx: ", to_str_safe(specific_audio_absolute_idx))

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

    -- Fallback logic if the initial FFmpeg mapping failed and no specific_audio_absolute_idx was initially found (e.g. target language not found)
    if (ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0) and not specific_audio_absolute_idx then
        log("warn", "FFmpeg (map '", ffmpeg_map_value_for_log ,"') failed. Trying fallback. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        local fallback_offset, fallback_idx = get_audio_stream_info(current_media_path, nil) -- nil for any audio
        if fallback_offset ~= audio_stream_offset_seconds then
            log("info", "Updating audio offset for fallback to: ", fallback_offset, "s.")
            audio_stream_offset_seconds = fallback_offset
        end
        if fallback_idx then
            ffmpeg_map_value_for_log = "0:" .. fallback_idx
        else
            ffmpeg_map_value_for_log = "0:a:0?"
            log("warn", "FFprobe found no audio for fallback index. Using '0:a:0?'.")
        end

        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_extract, v) end
        table.insert(ffmpeg_args_extract, temp_audio_raw_path)
        log("debug", "Running FFmpeg extract (fallback map '", ffmpeg_map_value_for_log, "'): ", table.concat(ffmpeg_args_extract, " "))
        ffmpeg_res_extract = utils.subprocess({ args = ffmpeg_args_extract, cancellable = false, capture_stdout = true, capture_stderr = true })
    end

    if ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then
        return validation_failed("FFmpeg audio extraction failed. Map '" .. ffmpeg_map_value_for_log .. "'. Stderr: " .. to_str_safe(ffmpeg_res_extract.stderr))
    end
    if not utils.file_info(temp_audio_raw_path) or utils.file_info(temp_audio_raw_path).size == 0 then
        return validation_failed("FFmpeg ran but raw temp audio '" .. temp_audio_raw_path .. "' not created or empty.")
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
            return validation_failed("FFmpeg audio filtering failed. Stderr: " .. to_str_safe(ffmpeg_res_filter.stderr))
        end
        if not utils.file_info(temp_audio_for_python) or utils.file_info(temp_audio_for_python).size == 0 then
            return validation_failed("FFmpeg filtering ran but final audio '" .. temp_audio_for_python .. "' missing or empty.")
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
        return validation_failed("Parakeet Python script execution failed. Error: " .. to_str_safe(python_res.error) .. ", Status: " .. to_str_safe(python_res.status) .. ". Stdout: " .. to_str_safe(python_res.stdout) .. ". Stderr: " .. to_str_safe(python_res.stderr))
    end

    log("info", "Python script finished. Status: ", to_str_safe(python_res.status), ". Stdout: ", to_str_safe(python_res.stdout), ". Stderr: ", to_str_safe(python_res.stderr))

    if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
        mp.commandv("sub-add", srt_output_path, "select")
        mp.osd_message("Parakeet: Subtitles loaded.", osd_duration_default)
        log("info", "SRT loaded: ", srt_output_path)
    else
        local srt_error_msg = "SRT file not found after Python script: " .. srt_output_path
        if utils.file_info(srt_output_path) then -- File exists but is empty
            srt_error_msg = "SRT file found but is empty: " .. srt_output_path
        end
        return validation_failed(srt_error_msg)
    end
    log("info", "Transcription process complete.")
end

--- Wrapper function to call `do_transcription_core` with default settings (no float32, no FFmpeg preprocessing).
local function transcribe_default_wrapper() do_transcription_core(false, false) end

--- Wrapper function to call `do_transcription_core` with Python float32 precision.
local function transcribe_py_float32_wrapper() do_transcription_core(true, false) end

--- Wrapper function to call `do_transcription_core` with FFmpeg pre-processing.
local function transcribe_ffmpeg_preprocess_wrapper() do_transcription_core(false, true) end

--- Wrapper function to call `do_transcription_core` with both FFmpeg pre-processing and Python float32 precision.
local function transcribe_ffmpeg_py_float32_wrapper() do_transcription_core(true, true) end

-- Register key bindings for the different transcription modes.
mp.add_key_binding(key_binding_default, "parakeet-transcribe-default", transcribe_default_wrapper)
mp.add_key_binding(key_binding_py_float32, "parakeet-transcribe-py-float32", transcribe_py_float32_wrapper)
mp.add_key_binding(key_binding_ffmpeg_preprocess, "parakeet-transcribe-ffmpeg-preprocess", transcribe_ffmpeg_preprocess_wrapper)
mp.add_key_binding(key_binding_ffmpeg_py_float32, "parakeet-transcribe-ffmpeg-py-float32", transcribe_ffmpeg_py_float32_wrapper)

log("info", "Parakeet (Reduced OSD, SRT in Temp) script loaded.")
log("info", "Keybindings: Default (" .. key_binding_default .. "), Float32 (" .. key_binding_py_float32 .. "), FFmpegPre (" .. key_binding_ffmpeg_preprocess .. "), FFmpegPre+Float32 (" .. key_binding_ffmpeg_py_float32 .. ")")
log("info", "Using Python: ", python_exe)
log("info", "Parakeet script: ", parakeet_script_path)
log("info", "Temp directory for audio and SRT files: ", temp_dir)
log("info", "FFmpeg filters: ", ffmpeg_audio_filters)
log("info", "Default OSD duration: " .. osd_duration_default .. "s")

--- Event handler for MPV's "shutdown" event.
-- Cleans up temporary files created by the script from `temp_dir`.
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

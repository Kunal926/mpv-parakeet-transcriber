-- Lua script for MPV to transcribe audio using parakeet_transcribe.py.
-- MODIFIED:
-- - Changed `sub-add` flag from "auto" to "select" for immediate subtitle activation.
-- - Hotkeys: Alt+4, Alt+5, Alt+6, Alt+7.
-- - Added comprehensive Lua-style docstrings and comments.
-- - SRT loading now attempted immediately after Python script finishes.
-- - Temporary file cleanup moved to MPV shutdown event.

local mp = require 'mp'
local utils = require 'mp.utils'

-- Subtitle alignment and styling are configured via mpv.conf;
-- this script intentionally avoids forcing ASS overrides.

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

--- Directory for storing temporary audio files.
-- This directory must exist and be writable by MPV and the user running MPV.
-- Temporary WAV files extracted from the media will be stored here before transcription.
-- These files are cleaned up when MPV shuts down.
-- @type string
-- @example "C:/temp_audio_mpv"
-- @example "/tmp/mpv_parakeet_audio"
local temp_dir = "C:/temp"

-- Keybindings for different transcription modes.
local key_binding_default = "Alt+4"             -- Standard transcription (no FFmpeg preprocessing, default Python precision)
local key_binding_py_float32 = "Alt+5"          -- Python Float32 precision (no FFmpeg preprocessing)
local key_binding_ffmpeg_preprocess = "Alt+6"   -- FFmpeg Preprocessing (default Python precision)
local key_binding_ffmpeg_py_float32 = "Alt+7" -- FFmpeg Preprocessing + Python Float32 Precision
local key_binding_isolate_asr_fast = "Alt+8"   -- Vocal isolation + ASR (fast)
local key_binding_isolate_asr_slow = "Alt+9"   -- Vocal isolation + ASR (high quality)

-- Root directory containing separation model weights.
-- Provide the full path so Python can locate the YAML and checkpoint files reliably.
local weights_dir = "C:/Parakeet_Caption/weights"

-- === Separation models you selected in A/B ===
local sep_fast = {
    cfg   = weights_dir .. "/roformer/voc_fv4/voc_gabox.yaml",
    ckpt  = weights_dir .. "/roformer/voc_fv4/voc_fv4.ckpt",
    target = "vocals"
}

local sep_slow = {
    cfg   = weights_dir .. "/roformer/karaoke_viperx/config_mel_band_roformer_karaoke.yaml",
    ckpt  = weights_dir .. "/roformer/karaoke_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    target = "vocals"
}

-- Word segmenter knobs (what you used in batch)
local seg_args = { "--segmenter","word","--max_words","12","--max_duration","6.0","--pause","0.6","--force_float32" }

--- FFmpeg audio filter chain for pre-processing mode.
-- This string defines the audio filters FFmpeg will apply when the
-- FFmpeg pre-processing mode is activated (e.g., via Alt+6 or Alt+7).
-- Filters can help improve transcription accuracy by normalizing volume, reducing noise, etc.
-- The filter chain should be a valid FFmpeg -af argument.
-- @type string
-- @example "loudnorm=I=-16:LRA=7:TP=-1.5" (Loudness normalization)
-- @example "anlmdn" (Audio Non-Local Means de-noiser)
-- @example "afftdn" (FFT-based de-noiser)
-- @example "loudnorm,anlmdn" (Apply multiple filters, comma-separated)
local ffmpeg_audio_filters = "loudnorm=I=-16:LRA=7:TP=-1.5"
-- ###################################

--- Table to store paths of temporary files for cleanup on MPV shutdown.
-- When temporary audio files are created, their paths are added to this table.
-- The `mp.register_event("shutdown", ...)` function iterates through this table
-- to remove these files when MPV closes.
-- @type table<string>
local files_to_cleanup_on_shutdown = {}

--- Flag to prevent concurrent transcription runs.
-- When true, additional transcription requests are ignored until completion.
-- @type boolean
local transcription_in_progress = false

--- Safely converts a value to its string representation.
-- Handles `nil` values by returning the string "nil", preventing errors
-- that would occur if `tostring(nil)` was called directly in concatenations.
-- @param val any The value to convert to a string.
-- @return string The string representation of the value, or "nil" if `val` is `nil`.
-- @usage local str_val = to_str_safe(my_variable)
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
-- Prefixes messages with "[parakeet_mpv]" and sends them to MPV's console
-- using the appropriate `mp.msg` level.
-- Additionally, it displays On-Screen Display (OSD) messages for "info", "warn",
-- and "error" levels to provide immediate user feedback.
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

    -- Display On-Screen Display (OSD) messages for user feedback
    if level == "error" then mp.osd_message("Parakeet Error: " .. message, 7) -- Duration 7 seconds
    elseif level == "warn" then mp.osd_message("Parakeet Warning: " .. message, 5) -- Duration 5 seconds
    elseif level == "info" then mp.osd_message("Parakeet: " .. message, 3) -- Duration 3 seconds
    end
end

--- Safely removes a file from the filesystem.
-- Checks if the file exists before attempting removal. Logs the action (success or failure)
-- using the `log` function at "debug" or "warn" level.
-- @param filepath string The path to the file to be removed. If `nil`, the function returns immediately.
-- @param description string (optional) A description of the file for logging purposes (e.g., "raw audio", "filtered audio"). Defaults to "unspecified".
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
-- @usage local start_time, stream_idx = get_audio_stream_info("/path/to/video.mkv", "eng")
-- @usage local start_time_any, stream_idx_any = get_audio_stream_info("/path/to/audio.mp3")
local function get_audio_stream_info(media_path, target_language_code)
    local ffprobe_cmd_args = {
        ffprobe_path,
        "-v", "quiet",           -- Suppress verbose output from ffprobe
        "-print_format", "json", -- Output in JSON format
        "-show_streams",         -- Get information about all streams
        "-select_streams", "a",  -- Select only audio streams
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
               stream.tags.language:lower():match("^"..target_language_code:lower()) then -- Match prefix for codes like "eng" vs "en"
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
-- This function orchestrates the entire transcription process:
-- 1. Validates necessary executables and directories.
-- 2. Retrieves the current media path and derives output/temporary file names.
-- 3. Uses `get_audio_stream_info` to determine the audio stream and its start offset, prioritizing English.
-- 4. Extracts the selected audio track to a temporary WAV file using FFmpeg. Includes fallback logic if the initial stream selection fails.
-- 5. If `apply_ffmpeg_filters_flag` is true, applies the configured `ffmpeg_audio_filters` to the extracted WAV file, creating another temporary WAV.
-- 6. Invokes the `parakeet_transcribe.py` script with the appropriate temporary audio file and parameters (including start offset and float32 flag).
-- 7. Attempts to load the generated SRT subtitle file into MPV using `sub-add` with the `select` flag to make it active.
-- 8. Logs progress and errors, displaying OSD messages for user feedback.
-- Temporary audio files created during this process are added to `files_to_cleanup_on_shutdown`.
--
-- @param force_python_float32_flag boolean If `true`, the "--force_float32" argument is passed to
-- the `parakeet_transcribe.py` script, potentially changing its processing precision.
-- @param apply_ffmpeg_filters_flag boolean If `true`, the audio extracted by FFmpeg is further
-- processed using the filter chain defined in the `ffmpeg_audio_filters` configuration variable
-- before being passed to the Python script.
-- @effects Creates temporary audio files in `temp_dir`.
-- @effects Creates an SRT subtitle file in the same directory as the media file.
-- @effects Loads the generated SRT file into MPV.
-- @usage Called by wrapper functions associated with keybindings (e.g., `transcribe_default_wrapper`).
local function do_transcription_core(force_python_float32_flag, apply_ffmpeg_filters_flag)
    if transcription_in_progress then
        mp.osd_message("Parakeet: Transcription already running.", osd_duration_default)
        return
    end
    transcription_in_progress = true
    local function abort()
        transcription_in_progress = false
    end
    -- Step 0: Validations
    if not utils.file_info(python_exe) then
        log("error", "Python executable not found: ", python_exe)
        abort()
        return
    end
    if not utils.file_info(parakeet_script_path) then
        log("error", "Parakeet Python script not found: '", parakeet_script_path, "'")
        abort()
        return
    end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then
        log("error", "FFmpeg executable not found: ", ffmpeg_path)
        abort()
        return
    end
    if ffprobe_path ~= "ffprobe" and not utils.file_info(ffprobe_path) then
        log("error", "FFprobe executable not found: ", ffprobe_path)
        abort()
        return
    end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory.")
        abort()
        return
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        log("error", "No media file is currently playing.")
        abort()
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

    -- Prioritize English audio streams for transcription.
    -- The "eng" code will try to match common English language tags (e.g., "eng", "en").
    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng")
    log("info", "Determined audio stream start offset: ", audio_stream_offset_seconds, "s. Specific stream absolute index for FFmpeg: ", to_str_safe(specific_audio_absolute_idx))

    -- Step 1: Extract audio track with FFmpeg
    mp.osd_message("Parakeet: Preparing audio with FFmpeg...", 5)
    log("info", "Step 1.1: Extracting audio track with FFmpeg...")
    log("info", "Outputting raw temporary audio to: ", temp_audio_raw_path)

    -- Common FFmpeg args: no video, 16 kHz mono via soxr resampler, float32 output, overwrite file
    local ffmpeg_common_args = {
        "-vn",
        "-ac", "1",
        "-af", "aresample=16000:resampler=soxr:precision=28",
        "-c:a", "pcm_f32le",
        "-y"
    }
    local ffmpeg_args_extract
    local ffmpeg_map_value_for_log -- For logging which -map option was used

    if specific_audio_absolute_idx then
        ffmpeg_map_value_for_log = "0:" .. specific_audio_absolute_idx -- Map by absolute stream index from ffprobe
        log("info", "Using specific stream map for FFmpeg extraction: ", ffmpeg_map_value_for_log)
        ffmpeg_args_extract = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    else
        -- If ffprobe didn't return a specific index (e.g., no 'eng' stream, or no audio streams at all),
        -- first try to let FFmpeg pick an English audio stream by metadata.
        log("warn", "No specific stream index from ffprobe (or target language 'eng' not found). Attempting FFmpeg default English mapping first for extraction.")
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
        -- This also updates the audio_stream_offset_seconds if the fallback stream has a different start time.
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
            abort()
            return
        end
    elseif ffmpeg_res_extract.error or ffmpeg_res_extract.status ~= 0 then
        -- This case handles failure of the initial attempt when specific_audio_absolute_idx was initially valid.
        log("error", "FFmpeg extraction with specific map '", ffmpeg_map_value_for_log, "' failed. Stderr: ", to_str_safe(ffmpeg_res_extract.stderr))
        mp.osd_message("Parakeet: Failed to extract audio with FFmpeg (specific map).", 7)
        abort()
        return
    end

    if not utils.file_info(temp_audio_raw_path) or utils.file_info(temp_audio_raw_path).size == 0 then
        log("error", "FFmpeg ran but raw temporary audio file '", temp_audio_raw_path, "' was not created or is empty.")
        mp.osd_message("Parakeet: FFmpeg failed to produce raw audio file.", 7)
        abort()
        return
    end
    log("info", "FFmpeg raw audio extraction successful: ", temp_audio_raw_path)

    if apply_ffmpeg_filters_flag then
        temp_audio_for_python = utils.join_path(temp_dir, sanitized_base_name .. "_audio_filtered.wav")
        if temp_audio_raw_path ~= temp_audio_for_python then
             table.insert(files_to_cleanup_on_shutdown, temp_audio_for_python)
        end
        log("info", "Step 1.5: Applying FFmpeg audio filters: ", ffmpeg_audio_filters)
        mp.osd_message("Parakeet: Applying FFmpeg audio filters...", 5)
        local filter_chain = ffmpeg_audio_filters
        if filter_chain ~= "" then
            filter_chain = filter_chain .. ",aresample=16000:resampler=soxr:precision=28"
        else
            filter_chain = "aresample=16000:resampler=soxr:precision=28"
        end
        local ffmpeg_args_filter = {
            ffmpeg_path,
            "-i", temp_audio_raw_path,
            "-af", filter_chain,
            "-ac", "1",
            "-c:a", "pcm_f32le",
            "-y",
            temp_audio_for_python
        }
        log("debug", "Running FFmpeg filter pass: ", table.concat(ffmpeg_args_filter, " "))
        local ffmpeg_res_filter = utils.subprocess({ args = ffmpeg_args_filter, cancellable = false, capture_stdout = true, capture_stderr = true })

        if ffmpeg_res_filter.error or ffmpeg_res_filter.status ~= 0 then
            log("error", "FFmpeg audio filtering failed. Error: ", to_str_safe(ffmpeg_res_filter.error), ", Status: ", to_str_safe(ffmpeg_res_filter.status))
            if ffmpeg_res_filter.stderr and string.len(ffmpeg_res_filter.stderr) > 0 then log("error", "FFmpeg Filter Stderr: ", ffmpeg_res_filter.stderr) end
            mp.osd_message("Parakeet: FFmpeg audio filtering failed.", 7)
            abort()
            return
        end
        if not utils.file_info(temp_audio_for_python) or utils.file_info(temp_audio_for_python).size == 0 then
            log("error", "FFmpeg filtering ran but final audio file '", temp_audio_for_python, "' is missing or empty.")
            mp.osd_message("Parakeet: FFmpeg filtering produced no audio.", 7)
            abort()
            return
        end
        log("info", "FFmpeg audio filtering successful. Final audio for transcription: ", temp_audio_for_python)
    end

    mp.osd_message("Parakeet (" .. mode_description .. "): Transcribing... This may take a while.", 7)
    log("info", "Step 2.1: Starting Parakeet transcription for: ", file_name_with_ext, " (Using audio: ", temp_audio_for_python, ")")

    local python_command_args = {
        python_exe,
        parakeet_script_path,
        temp_audio_for_python, -- Input WAV file
        srt_output_path,       -- Output SRT file path
        "--audio_start_offset", tostring(audio_stream_offset_seconds) -- Pass the determined start offset
    }
    if force_python_float32_flag then
        table.insert(python_command_args, "--force_float32")
    end

    for _, v in ipairs(seg_args) do table.insert(python_command_args, v) end

    local fps = mp.get_property_native("container-fps") or mp.get_property_native("fps") or 24
    table.insert(python_command_args, "--fps=" .. string.format("%.3f", fps))

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
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
             log("warn", "Python script exited with an error. Status: ", to_str_safe(python_res.status), ". Check Python script's own logging for details.")
             mp.osd_message("Parakeet: Python script error. Check console.", 7)
        end

        log("info", "Attempting to load SRT immediately: ", srt_output_path)
        if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
            mp.commandv("sub-add", srt_output_path, "select") -- Use "select" to force MPV to switch to this subtitle
            mp.osd_message("Parakeet: Loaded " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 3)
        elseif utils.file_info(srt_output_path) then
            log("warn", "SRT file found but is empty: ", srt_output_path, ". This might indicate a transcription problem or an error SRT with no content.")
            mp.osd_message("Parakeet: SRT file empty. Check console.", 5)
        else
            log("warn", "SRT file not found after Python script execution: ", srt_output_path, ". Transcription may have failed.")
            mp.osd_message("Parakeet: SRT not found. Check Python logs.", 7)
        end
    end
    log("info", "Transcription process complete. Temporary audio files (if any) will be cleaned on MPV shutdown.")
    abort()
end

-- Perform FFmpeg extraction -> RoFormer separation -> Parakeet ASR
local function run_isolate_then_asr(model)
    model = model or sep_fast
    if transcription_in_progress then
        mp.osd_message("Parakeet: Transcription already running.", osd_duration_default)
        return
    end
    transcription_in_progress = true
    local function abort()
        transcription_in_progress = false
    end

    if not utils.file_info(python_exe) then
        log("error", "Python executable not found: ", python_exe)
        abort()
        return
    end
    if not utils.file_info(parakeet_script_path) then
        log("error", "Parakeet Python script not found: '", parakeet_script_path, "'")
        abort()
        return
    end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then
        log("error", "FFmpeg executable not found: ", ffmpeg_path)
        abort()
        return
    end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory.")
        abort()
        return
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        log("error", "No media file is currently playing.")
        abort()
        return
    end

    local file_name_with_ext = current_media_path:match("([^/\\]+)$") or "unknown_file"
    local base_name = file_name_with_ext:match("(.+)%.[^%.]+$") or file_name_with_ext
    local media_dir = ""
    local path_sep_pos = current_media_path:match("^.*[/\\]()")
    if path_sep_pos then
        media_dir = current_media_path:sub(1, path_sep_pos -1)
    else
        local cwd = utils.getcwd()
        if cwd then media_dir = cwd end
    end
    if media_dir ~= "" and not media_dir:match("[/\\]$") then
        media_dir = media_dir .. package.config:sub(1,1)
    end
    local srt_output_path = utils.join_path(media_dir, base_name .. ".srt")
    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_")

    -- temp_stereo retains source sample rate (no pre-resample)
    local temp_stereo = utils.join_path(temp_dir, sanitized_base_name .. "_stereo_44k.wav")
    local temp_vocals_44k = utils.join_path(temp_dir, sanitized_base_name .. "_vocals_44k.wav")
    local temp_vocals_16k = utils.join_path(temp_dir, sanitized_base_name .. "_vocals_16k.wav")
    table.insert(files_to_cleanup_on_shutdown, temp_stereo)
    table.insert(files_to_cleanup_on_shutdown, temp_vocals_44k)
    table.insert(files_to_cleanup_on_shutdown, temp_vocals_16k)

    local audio_offset_seconds, audio_stream_idx = get_audio_stream_info(current_media_path, "eng")
    if not audio_offset_seconds then audio_offset_seconds = 0.0 end

    mp.osd_message("Extracting audio...", 3)
    log("info", "Step A: Extracting audio to ", temp_stereo)

    local map_arg
    if audio_stream_idx then
        map_arg = "0:" .. audio_stream_idx .. "?"
    else
        map_arg = "0:a:0?"
    end

    -- Extract stereo 44.1 kHz float32 to match ZFTurbo models exactly
    local ffmpeg_args = {ffmpeg_path, "-y", "-i", current_media_path, "-map", map_arg,
        "-ac", "2",
        "-af", "aresample=44100:resampler=soxr:precision=28",
        "-c:a", "pcm_f32le", "-vn", temp_stereo} -- stereo_44k float32
    local ffmpeg_res = utils.subprocess({ args = ffmpeg_args, cancellable = false, capture_stdout = true, capture_stderr = true })
    if ffmpeg_res.error or ffmpeg_res.status ~= 0 then
        log("warn", "FFmpeg extraction (" .. to_str_safe(map_arg) .. ") failed: ", to_str_safe(ffmpeg_res.stderr))
        ffmpeg_args = {ffmpeg_path, "-y", "-i", current_media_path, "-map", "0:a:0?",
            "-ac", "2",
            "-af", "aresample=44100:resampler=soxr:precision=28",
            "-c:a", "pcm_f32le", "-vn", temp_stereo}
        ffmpeg_res = utils.subprocess({ args = ffmpeg_args, cancellable = false, capture_stdout = true, capture_stderr = true })
        if ffmpeg_res.error or ffmpeg_res.status ~= 0 then
            log("error", "FFmpeg extraction failed: " , to_str_safe(ffmpeg_res.stderr))
            mp.osd_message("Parakeet: FFmpeg extraction failed.", 7)
            abort()
            return
        end
    end
    if not utils.file_info(temp_stereo) or utils.file_info(temp_stereo).size == 0 then
        log("error", "FFmpeg produced no audio.")
        mp.osd_message("Parakeet: FFmpeg produced no audio.", 7)
        abort()
        return
    end

    mp.osd_message("Separating vocals...", 5)
    log("info", "Step B: Separating vocals using model cfg ", model.cfg)
    local script_dir = utils.split_path(parakeet_script_path)
    local sep_script
    if script_dir ~= "" then
        sep_script = utils.join_path(script_dir, "separation")
        sep_script = utils.join_path(sep_script, "bsr_separate.py")
    else
        sep_script = utils.join_path("separation", "bsr_separate.py")
    end
    local t0 = mp.get_time()
    -- IMPORTANT: write separation result at 44.1 kHz stereo (match inference.py behavior)
    local sep_cmd = {
        python_exe, sep_script,
        "--in_wav", temp_stereo,
        "--out_wav", temp_vocals_44k,
        "--cfg", model.cfg, "--ckpt", model.ckpt, "--target", model.target,
        "--device", "cuda",
        "--fp16",
        "--save_sr", "44100",
        "--channels", "2"
    }
    log("info", "SEP CMD: " .. table.concat(sep_cmd, " "))
    local sep_res = utils.subprocess({ args = sep_cmd, cancellable = false, capture_stdout = true, capture_stderr = true })
    local dt = mp.get_time() - t0
    log("info", ("SEP DONE in %.1fs, rc=%s"):format(dt, tostring(sep_res.status)))
    if dt < 10 then
        log("warn", "Separator finished suspiciously fast — model likely didn’t run. See command above.")
    end
    if sep_res.error or sep_res.status ~= 0 then
        log("error", "Separation failed: ", to_str_safe(sep_res.stderr))
        mp.osd_message("Parakeet: Separation failed.", 7)
        abort()
        return
    end
    if not utils.file_info(temp_vocals_44k) or utils.file_info(temp_vocals_44k).size == 0 then
        log("error", "Separator produced no output")
        mp.osd_message("Parakeet: Separation produced no audio.", 7)
        abort()
        return
    end

    -- Downsample to 16 kHz mono float32 (soxr), exactly like batch_parakeet_srt.py
    mp.osd_message("Preparing 16 kHz mono for ASR...", 3)
    local ds_cmd = {
        ffmpeg_path, "-y",
        "-i", temp_vocals_44k,
        "-ac", "1",
        "-af", "aresample=resampler=soxr:precision=28",
        "-ar", "16000",
        "-c:a", "pcm_f32le",
        temp_vocals_16k
    }
    local ds_res = utils.subprocess({ args = ds_cmd, cancellable = false, capture_stdout = true, capture_stderr = true })
    if ds_res.error or ds_res.status ~= 0 or (not utils.file_info(temp_vocals_16k)) or (utils.file_info(temp_vocals_16k).size == 0) then
        log("error", "Downsample step failed. Stderr: ", to_str_safe(ds_res.stderr))
        mp.osd_message("Parakeet: 16 kHz prep failed.", 7)
        abort()
        return
    end

    mp.osd_message("Transcribing...", 5)
    log("info", "Step C: Running Parakeet transcription on separated vocals")
    local parakeet_args = {
        python_exe, parakeet_script_path, temp_vocals_16k, srt_output_path,
        "--audio_start_offset", tostring(audio_offset_seconds)
    }
    for _,v in ipairs(seg_args) do table.insert(parakeet_args, v) end
    local fps = mp.get_property_native("container-fps") or mp.get_property_native("fps") or 24
    table.insert(parakeet_args, "--fps=" .. string.format("%.3f", fps))
    local python_opts = { args = parakeet_args, cancellable = false, capture_stdout = true, capture_stderr = true }
    local python_res = utils.subprocess(python_opts)
    if python_res.error then
        log("error", "Failed to launch Parakeet Python script: ", to_str_safe(python_res.error))
        mp.osd_message("Parakeet: Failed to launch Python.", 7)
    else
        if python_res.stderr and string.len(python_res.stderr) > 0 then log("debug", "Python stderr: ", python_res.stderr) end
        if python_res.status ~= nil and python_res.status ~= 0 then
            mp.osd_message("Parakeet: Python script error.", 7)
        end
        if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
            mp.commandv("sub-add", srt_output_path, "select")
            mp.osd_message("Parakeet: Loaded SRT", 3)
        else
            mp.osd_message("Parakeet: SRT not found.", 7)
        end
    end

    abort()
end

--- Wrapper function to call `do_transcription_core` with default settings.
-- This function is bound to the `key_binding_default` hotkey.
-- It invokes transcription without forcing Python float32 precision and without FFmpeg pre-processing.
local function transcribe_default_wrapper()
    do_transcription_core(false, false)
end

--- Wrapper function to call `do_transcription_core` with Python float32 precision.
-- This function is bound to the `key_binding_py_float32` hotkey.
-- It invokes transcription forcing Python float32 precision and without FFmpeg pre-processing.
local function transcribe_py_float32_wrapper()
    do_transcription_core(true, false)
end

--- Wrapper function to call `do_transcription_core` with FFmpeg pre-processing.
-- This function is bound to the `key_binding_ffmpeg_preprocess` hotkey.
-- It invokes transcription without forcing Python float32 precision but with FFmpeg pre-processing
-- using the filters defined in `ffmpeg_audio_filters`.
local function transcribe_ffmpeg_preprocess_wrapper()
    do_transcription_core(false, true)
end

--- Wrapper function to call `do_transcription_core` with both FFmpeg pre-processing and Python float32 precision.
-- This function is bound to the `key_binding_ffmpeg_py_float32` hotkey.
-- It invokes transcription forcing Python float32 precision and with FFmpeg pre-processing
-- using the filters defined in `ffmpeg_audio_filters`.
local function transcribe_ffmpeg_py_float32_wrapper()
    do_transcription_core(true, true)
end

-- Register key bindings for the different transcription modes.
-- Each binding maps a key combination to its respective wrapper function.
mp.add_key_binding(key_binding_default, "parakeet-transcribe-default", transcribe_default_wrapper)
mp.add_key_binding(key_binding_py_float32, "parakeet-transcribe-py-float32", transcribe_py_float32_wrapper)
mp.add_key_binding(key_binding_ffmpeg_preprocess, "parakeet-transcribe-ffmpeg-preprocess", transcribe_ffmpeg_preprocess_wrapper)
mp.add_key_binding(key_binding_ffmpeg_py_float32, "parakeet-transcribe-ffmpeg-py-float32", transcribe_ffmpeg_py_float32_wrapper)
-- Alt+8 fast = fv4
mp.add_forced_key_binding(key_binding_isolate_asr_fast, "parakeet_fast", function() run_isolate_then_asr(sep_fast) end)
-- Alt+9 slow = mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956
mp.add_forced_key_binding(key_binding_isolate_asr_slow, "parakeet_slow", function() run_isolate_then_asr(sep_slow) end)

log("info", "Parakeet (Multi-Mode) script loaded.")
log("info", "SRT will be loaded immediately after transcription and selected.")
log("info", "Temporary files will be cleaned up on MPV shutdown.")
log("info", "Press '", key_binding_default, "' for Standard Transcription.")
log("info", "Press '", key_binding_py_float32, "' for Python Float32 Precision.")
log("info", "Press '", key_binding_ffmpeg_preprocess, "' for FFmpeg Preprocessing (Default Python Precision).")
log("info", "Press '", key_binding_ffmpeg_py_float32, "' for FFmpeg Preprocessing + Python Float32 Precision.")
log("info", "Press '", key_binding_isolate_asr_fast, "' for Vocal Isolation + ASR (fast).")
log("info", "Press '", key_binding_isolate_asr_slow, "' for Vocal Isolation + ASR (high quality).")
log("info", "Using Python from: ", python_exe)
log("info", "Using FFmpeg from: ", ffmpeg_path)
log("info", "Using FFprobe from: ", ffprobe_path)
log("info", "Parakeet script: ", parakeet_script_path)
log("info", "Temporary file directory: ", temp_dir)
log("info", "FFmpeg pre-processing filters: ", ffmpeg_audio_filters)

--- Event handler for MPV's "shutdown" event.
-- This function is called when MPV is closing. It iterates through the
-- `files_to_cleanup_on_shutdown` table and attempts to remove each file
-- listed, using `safe_remove` to handle errors and logging.
-- After attempting cleanup, the `files_to_cleanup_on_shutdown` table is cleared.
mp.register_event("shutdown", function()
    log("info", "MPV shutdown event. Cleaning up temporary Parakeet files...")
    if #files_to_cleanup_on_shutdown > 0 then
        for _, filepath in ipairs(files_to_cleanup_on_shutdown) do
            safe_remove(filepath, "Shutdown cleanup")
        end
        files_to_cleanup_on_shutdown = {} -- Clear the table after attempting cleanup
    else
        log("info", "No temporary files registered for cleanup.")
    end
    log("info", "Parakeet shutdown cleanup finished.")
end)

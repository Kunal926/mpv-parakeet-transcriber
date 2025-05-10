-- parakeet_mpv_ffmpeg_venv.lua (v8b - Corrected FFmpeg Map)
-- Lua script for MPV to transcribe the ENTIRE audio file using parakeet_transcribe.py
-- from a specified Python virtual environment, after pre-processing with FFmpeg
-- to select an English audio track.
-- NEW: Detects audio stream start_time offset using ffprobe and passes it to Python.
-- FIX: Uses correct -map 0:index syntax when specific_audio_stream_idx is found.

local mp = require 'mp'
local utils = require 'mp.utils'

-- ########## Configuration ##########
-- 1. Path to your Python executable WITHIN your virtual environment.
local python_exe = "C:/venvs/nemo_mpv_py312/Scripts/python.exe"

-- 2. Path to the parakeet_transcribe.py script.
local parakeet_script_path = "C:/Parakeet_Caption/parakeet_transcribe.py" -- Hardcoded as per user request

-- 3. Full path to your FFmpeg and FFprobe executables.
--    If "ffmpeg"/"ffprobe", it's assumed to be in system PATH. Otherwise, provide full path.
local ffmpeg_path = "ffmpeg"
local ffprobe_path = "ffprobe" -- Added for ffprobe

-- 4. Path to a directory for temporary files.
local temp_dir = "C:/temp"

-- 5. Keybinding to trigger the transcription.
local key_binding = "ctrl+w"

-- 6. Delay in seconds before attempting to automatically load the generated SRT file
--    AND clean up the temporary audio file.
local auto_load_and_cleanup_delay_seconds = 30

-- ###################################

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

    if level == "error" then
        mp.osd_message("Parakeet Error: " .. message, 7)
    elseif level == "warn" then
        mp.osd_message("Parakeet Warning: " .. message, 5)
    elseif level == "info" then
        mp.osd_message("Parakeet: " .. message, 3)
    end
end

-- Function to safely remove a file
local function safe_remove(filepath, description)
    if not filepath then return end
    if utils.file_info(filepath) then
        local success, err_msg = os.remove(filepath)
        if success then
            log("debug", "Cleaned up temporary file (" .. (description or "") .. "): ", filepath)
        else
            log("warn", "Failed to remove temporary file (" .. (description or "") .. "): ", filepath, " - Error: ", (err_msg or "unknown"))
        end
    else
        log("debug", "Temporary file (" .. (description or "") .. ") not found for removal, skipping: ", filepath)
    end
end

-- NEW: Function to get audio stream start offset and specific index
local function get_audio_stream_info(media_path, target_language_code)
    local ffprobe_cmd_args = {
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a", -- Select all audio streams
        media_path
    }
    log("debug", "Running ffprobe to find audio streams: ", table.concat(ffprobe_cmd_args, " "))
    local res = utils.subprocess({args = ffprobe_cmd_args, cancellable = false})

    if res.error or res.status ~= 0 or not res.stdout then
        log("warn", "ffprobe command failed or produced no output. Error: ", to_str_safe(res.error), ", Status: ", to_str_safe(res.status))
        if res.stderr and string.len(res.stderr) > 0 then log("warn", "ffprobe stderr: ", res.stderr) end
        return 0.0, nil -- Default to 0.0 offset, no specific stream index found
    end

    local success, data = pcall(utils.parse_json, res.stdout)
    if not success or not data or not data.streams or #data.streams == 0 then
        log("warn", "Failed to parse ffprobe JSON output or no audio streams found. Data: ", to_str_safe(res.stdout))
        return 0.0, nil
    end

    local selected_stream_absolute_index = nil -- This will be the stream index like "0", "1", "2" from ffprobe
    local selected_stream_start_time = 0.0

    -- Try to find the target language stream
    if target_language_code then
        for _, stream in ipairs(data.streams) do
            if stream.codec_type == "audio" and stream.tags and stream.tags.language and stream.tags.language:lower():match(target_language_code:lower()) then
                selected_stream_absolute_index = tostring(stream.index) -- FFprobe's absolute stream index
                selected_stream_start_time = tonumber(stream.start_time) or 0.0
                log("info", "Found target language stream '", target_language_code, "' with absolute index ", selected_stream_absolute_index, " and start_time ", selected_stream_start_time)
                return selected_stream_start_time, selected_stream_absolute_index
            end
        end
        log("warn", "Target language '", target_language_code, "' not found in ffprobe metadata tags.")
    end

    -- If target language not found (or not specified), use the first audio stream as a fallback for offset and index
    log("info", "No specific language match or no target language. Using first audio stream for offset and index.")
    for _, stream in ipairs(data.streams) do -- Iterate again to find the first audio stream
        if stream.codec_type == "audio" then
            selected_stream_absolute_index = tostring(stream.index)
            selected_stream_start_time = tonumber(stream.start_time) or 0.0
            log("info", "Using first available audio stream (absolute index ", selected_stream_absolute_index, ") with start_time ", selected_stream_start_time, " as reference for offset and mapping.")
            return selected_stream_start_time, selected_stream_absolute_index
        end
    end
    
    log("warn", "No audio streams found at all by ffprobe.")
    return 0.0, nil 
end


local function transcribe_current_file()
    -- Validations (same as before)
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
    if media_dir ~= "" and not media_dir:match("[/\\]$") then media_dir = media_dir .. package.config:sub(1,1) end

    local srt_output_path = utils.join_path(media_dir, base_name .. ".srt")
    local sanitized_base_name = base_name:gsub("[^%w%-_%.]", "_") 
    local temp_audio_path = utils.join_path(temp_dir, sanitized_base_name .. "_audio_for_transcription.wav")

    mp.osd_message("Parakeet: Analyzing audio streams...", 3)
    log("info", "Step 0: Analyzing audio streams with FFprobe...")
    local audio_stream_offset_seconds, specific_audio_absolute_idx = get_audio_stream_info(current_media_path, "eng")
    log("info", "Determined audio stream start offset: ", audio_stream_offset_seconds, "s. Specific stream absolute index for FFmpeg: ", to_str_safe(specific_audio_absolute_idx))

    mp.osd_message("Parakeet: Preparing audio with FFmpeg...", 5)
    log("info", "Step 1: Extracting audio track with FFmpeg...")
    log("info", "Outputting temporary audio to: ", temp_audio_path)

    local ffmpeg_common_args = {"-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y"}
    local ffmpeg_args
    local ffmpeg_map_value_for_log -- For logging the map value used

    if specific_audio_absolute_idx then
        -- If ffprobe found a specific English (or first) stream index, use its absolute index directly
        ffmpeg_map_value_for_log = "0:" .. specific_audio_absolute_idx -- CORRECTED MAP
        log("info", "Using specific stream map for FFmpeg: ", ffmpeg_map_value_for_log)
        ffmpeg_args = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    else
        log("warn", "No specific stream index from ffprobe. Attempting FFmpeg default English mapping first.")
        ffmpeg_map_value_for_log = "0:a:m:language:eng"
        ffmpeg_args = {ffmpeg_path, "-i", current_media_path, "-map", ffmpeg_map_value_for_log}
    end
    
    for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args, v) end
    table.insert(ffmpeg_args, temp_audio_path)
    
    log("debug", "Running FFmpeg with map '", ffmpeg_map_value_for_log, "': ", table.concat(ffmpeg_args, " "))
    local ffmpeg_res = utils.subprocess({ args = ffmpeg_args, cancellable = false })

    -- If the primary attempt failed AND we didn't initially have a specific index (meaning we tried language map)
    if (ffmpeg_res.error or ffmpeg_res.status ~= 0) and not specific_audio_absolute_idx then
        log("warn", "FFmpeg (map '", ffmpeg_map_value_for_log ,"') command failed. Error: ", to_str_safe(ffmpeg_res.error), ", Status: ", to_str_safe(ffmpeg_res.status), ", Stderr: ", to_str_safe(ffmpeg_res.stderr))
        log("warn", "Attempting FFmpeg fallback to map first audio stream explicitly (0:a:0?)...")
        mp.osd_message("Parakeet: Specific/English audio map failed, trying fallback map 0:a:0?...", 3)
        
        -- Re-check offset for the first audio stream if we are explicitly falling back to 0:a:0?
        local fallback_offset_for_0a0, _ = get_audio_stream_info(current_media_path, nil) 
        if fallback_offset_for_0a0 ~= audio_stream_offset_seconds then
            log("info", "Updating audio offset for 0:a:0? fallback to: ", fallback_offset_for_0a0, "s.")
            audio_stream_offset_seconds = fallback_offset_for_0a0
        end
        ffmpeg_map_value_for_log = "0:a:0?" -- For logging
        ffmpeg_args = {ffmpeg_path, "-i", current_media_path, "-map", "0:a:0?"}
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args, v) end
        table.insert(ffmpeg_args, temp_audio_path)

        log("debug", "Running FFmpeg (fallback map 0:a:0?): ", table.concat(ffmpeg_args, " "))
        ffmpeg_res = utils.subprocess({ args = ffmpeg_args, cancellable = false }) 

        if ffmpeg_res.error or ffmpeg_res.status ~= 0 then
            log("error", "FFmpeg fallback audio extraction (0:a:0?) also failed. Error: ", to_str_safe(ffmpeg_res.error), ", Status: ", to_str_safe(ffmpeg_res.status), ", Stderr: ", to_str_safe(ffmpeg_res.stderr))
            mp.osd_message("Parakeet: Failed to extract audio with FFmpeg.", 7)
            safe_remove(temp_audio_path, "Failed FFmpeg temp audio")
            return
        end
    elseif ffmpeg_res.error or ffmpeg_res.status ~= 0 then -- If specific index attempt failed
        log("error", "FFmpeg with specific map '", ffmpeg_map_value_for_log, "' failed. Error: ", to_str_safe(ffmpeg_res.error), ", Status: ", to_str_safe(ffmpeg_res.status), ", Stderr: ", to_str_safe(ffmpeg_res.stderr))
        mp.osd_message("Parakeet: Failed to extract audio with FFmpeg (specific map).", 7)
        safe_remove(temp_audio_path, "Failed FFmpeg temp audio")
        return
    end
    
    if not utils.file_info(temp_audio_path) or utils.file_info(temp_audio_path).size == 0 then
        log("error", "FFmpeg ran but temporary audio file '", temp_audio_path, "' was not created or is empty.")
        mp.osd_message("Parakeet: FFmpeg failed to produce audio file.", 7)
        safe_remove(temp_audio_path, "Empty/missing FFmpeg temp audio")
        return
    end
    log("info", "FFmpeg audio extraction seems successful. Temporary audio at: ", temp_audio_path)

    mp.osd_message("Parakeet: Transcribing extracted audio... This may take a while.", 7)
    log("info", "Step 2: Starting Parakeet transcription for: ", file_name_with_ext)

    local python_command_args = {
        python_exe,
        parakeet_script_path,
        temp_audio_path,    
        srt_output_path,
        "--audio_start_offset", tostring(audio_stream_offset_seconds) 
    }

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
    local python_res = utils.subprocess({ args = python_command_args, cancellable = false })

    if python_res.error then 
        log("error", "Failed to launch Parakeet Python script: ", (python_res.error or "Unknown error"))
        if python_res.stderr and string.len(python_res.stderr) > 0 then
             log("error", "Stderr from Python launch failure: ", python_res.stderr)
        end
        safe_remove(temp_audio_path, "Temp audio after Python launch fail")
        return
    end

    log("info", "Parakeet Python script launched (PID: ", (python_res.pid or "unknown"), "). Transcription in progress...")
    
    if auto_load_and_cleanup_delay_seconds > 0 then
        mp.add_timeout(auto_load_and_cleanup_delay_seconds, function()
            if python_res.status ~= nil and python_res.status ~= 0 then
                 log("warn", "Python script may have exited with an error. Status: ", to_str_safe(python_res.status), ". Stderr: ", to_str_safe(python_res.stderr))
            end

            log("info", "Attempting to load SRT: ", srt_output_path)
            if utils.file_info(srt_output_path) and utils.file_info(srt_output_path).size > 0 then
                mp.commandv("sub-add", srt_output_path, "auto")
                mp.osd_message("Parakeet: Attempted to load " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 3)
            elseif utils.file_info(srt_output_path) then -- File exists but is empty
                log("warn", "SRT file found but is empty: ", srt_output_path)
                mp.osd_message("Parakeet: SRT file empty. Check console.", 5)
            else
                log("warn", "SRT file not found after delay: ", srt_output_path)
                mp.osd_message("Parakeet: SRT not found. Check console for Python errors.", 7)
            end
            log("info", "Cleaning up temporary audio file: ", temp_audio_path)
            safe_remove(temp_audio_path, "Delayed cleanup of temp audio")
        end)
    else
        log("info", "Auto-load/cleanup disabled. Temporary audio file may remain: ", temp_audio_path)
        mp.osd_message("Parakeet: Transcription initiated. SRT: " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 7)
    end
end

mp.add_key_binding(key_binding, "parakeet-transcribe-ffmpeg", transcribe_current_file)

log("info", "Parakeet (FFmpeg, Offset Adjust) script loaded. Press '", key_binding, "' to transcribe.")
log("info", "Using Python from: ", python_exe)
log("info", "Using FFmpeg from: ", ffmpeg_path)
log("info", "Using FFprobe from: ", ffprobe_path)
log("info", "Parakeet script (hardcoded): ", parakeet_script_path) 
log("info", "Temporary file directory: ", temp_dir)

mp.register_event("shutdown", function()
    log("info", "MPV shutdown: Checking for leftover Parakeet temporary audio files...")
end)

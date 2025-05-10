-- parakeet_mpv_ffmpeg_venv.lua (v7 - Improved OSD messages for clarity)
-- Lua script for MPV to transcribe the ENTIRE audio file using parakeet_transcribe.py
-- from a specified Python virtual environment, after pre-processing with FFmpeg
-- to select an English audio track.

local mp = require 'mp'
local utils = require 'mp.utils'

-- ########## Configuration ##########
-- 1. Path to your Python executable WITHIN your virtual environment.
local python_exe = "C:/venvs/nemo_mpv_py312/Scripts/python.exe"

-- 2. Path to the parakeet_transcribe.py script.
local parakeet_script_path = "C:/Parakeet_Caption/parakeet_transcribe.py" -- Hardcoded as per user request

-- 3. Full path to your FFmpeg executable.
--    If "ffmpeg", it's assumed to be in system PATH. Otherwise, provide full path.
local ffmpeg_path = "ffmpeg"

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

-- Ensure temp directory exists (basic check)
if not utils.file_info(temp_dir) then
    mp.msg.warn("[parakeet_mpv] Temporary directory does not exist: " .. temp_dir)
    mp.msg.warn("[parakeet_mpv] Please create '" .. temp_dir .. "' manually if issues occur, or check permissions.")
    local mkdir_cmd
    if package.config:sub(1,1) == '\\' then -- Likely Windows
        if temp_dir:match("^[A-Za-z]:$") then
             mp.msg.warn("[parakeet_mpv] Cannot 'mkdir' a root drive like '" .. temp_dir .. "'. Please ensure it's accessible.")
        else
            mkdir_cmd = string.format('cmd /C "if not exist "%s" mkdir "%s""', temp_dir:gsub("/", "\\"), temp_dir:gsub("/", "\\"))
            local ok, err_code = os.execute(mkdir_cmd)
            if err_code == 0 then 
                 mp.msg.info("[parakeet_mpv] Attempted to create temp directory: " .. temp_dir .. " (Success or already exists)")
            else
                 mp.msg.warn("[parakeet_mpv] Failed to create temp directory (or command failed). Exit Code: " .. to_str_safe(err_code) .. ". Command: " .. mkdir_cmd)
            end
        end
    else -- Likely Linux/macOS
        mkdir_cmd = string.format('mkdir -p "%s"', temp_dir)
        local ok, err = os.execute(mkdir_cmd)
        if ok then mp.msg.info("[parakeet_mpv] Attempted to create temp directory: " .. temp_dir)
        else mp.msg.warn("[parakeet_mpv] Failed to attempt temp directory creation: " .. (err or "unknown")) end
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
    local file_info_check = utils.file_info(filepath)
    if file_info_check then
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

local function transcribe_current_file()
    -- Validations
    if not utils.file_info(python_exe) then
         log("error", "Python executable not found at configured venv path: ", python_exe, ". Please verify 'python_exe' and ensure the virtual environment is correctly set up.")
         return
    end
    if not utils.file_info(parakeet_script_path) then
        log("error", "parakeet_transcribe.py not found at hardcoded path: '", parakeet_script_path, "'. Please ensure the Python script is correctly located at this path.")
        return
    end
    if ffmpeg_path ~= "ffmpeg" and not utils.file_info(ffmpeg_path) then
        log("error", "FFmpeg executable not found at specified full path: ", ffmpeg_path, ". Please verify 'ffmpeg_path' or set to 'ffmpeg' if it's in PATH.")
        return
    end
    if not utils.file_info(temp_dir) or not utils.file_info(temp_dir).is_dir then
        log("error", "Temporary directory '", temp_dir, "' does not exist or is not a directory. Please create it and ensure permissions.")
        return
    end

    local current_media_path = mp.get_property_native("path")
    if not current_media_path or current_media_path == "" then
        log("error", "No file is currently playing.")
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
    local temp_audio_path = utils.join_path(temp_dir, sanitized_base_name .. "_eng_audio.wav")

    -- Initial OSD message before FFmpeg processing
    mp.osd_message("Parakeet: Preparing audio with FFmpeg...", 5) -- MODIFIED OSD Message

    log("info", "Step 1: Extracting audio track with FFmpeg...")
    log("info", "Outputting temporary audio to: ", temp_audio_path)

    local ffmpeg_common_args = {
        "-vn", 
        "-acodec", "pcm_s16le", 
        "-ar", "16000",        
        "-ac", "1",            
        "-y"                   
    }

    -- Attempt 1: Strictly map English audio track
    local ffmpeg_args_eng = {
        ffmpeg_path,
        "-i", current_media_path,
        "-map", "0:a:m:language:eng" 
    }
    for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_eng, v) end
    table.insert(ffmpeg_args_eng, temp_audio_path)
    
    log("debug", "Running FFmpeg (Strict English map): ", table.concat(ffmpeg_args_eng, " "))
    local ffmpeg_res_eng = utils.subprocess({ args = ffmpeg_args_eng, cancellable = false })

    if ffmpeg_res_eng.error or ffmpeg_res_eng.status ~= 0 then
        log("warn", "FFmpeg (Strict English map) command failed. Error: ", to_str_safe(ffmpeg_res_eng.error), ", Status: ", to_str_safe(ffmpeg_res_eng.status), ", Stderr: ", to_str_safe(ffmpeg_res_eng.stderr))
        log("warn", "Attempting FFmpeg fallback to first audio stream...")
        mp.osd_message("Parakeet: English audio not found, trying fallback...", 3) -- OSD for fallback
        
        local ffmpeg_args_fallback = {
            ffmpeg_path,
            "-i", current_media_path,
            "-map", "0:a:0?" 
        }
        for _, v in ipairs(ffmpeg_common_args) do table.insert(ffmpeg_args_fallback, v) end
        table.insert(ffmpeg_args_fallback, temp_audio_path)

        log("debug", "Running FFmpeg (fallback map): ", table.concat(ffmpeg_args_fallback, " "))
        local ffmpeg_res_fallback = utils.subprocess({ args = ffmpeg_args_fallback, cancellable = false })

        if ffmpeg_res_fallback.error or ffmpeg_res_fallback.status ~= 0 then
            log("error", "FFmpeg fallback audio extraction also failed. Error: ", to_str_safe(ffmpeg_res_fallback.error), ", Status: ", to_str_safe(ffmpeg_res_fallback.status), ", Stderr: ", to_str_safe(ffmpeg_res_fallback.stderr))
            mp.osd_message("Parakeet: Failed to extract audio with FFmpeg.", 7)
            safe_remove(temp_audio_path, "Failed FFmpeg temp audio")
            return
        end
    end
    
    local temp_audio_file_info = utils.file_info(temp_audio_path)
    if not temp_audio_file_info or temp_audio_file_info.size == 0 then
        log("error", "FFmpeg ran but temporary audio file '", temp_audio_path, "' was not created or is empty. This can happen if no suitable audio track was found by any FFmpeg attempt. Check FFmpeg logs in console.")
        mp.osd_message("Parakeet: FFmpeg failed to produce audio file.", 7)
        safe_remove(temp_audio_path, "Empty/missing FFmpeg temp audio")
        return
    end
    log("info", "FFmpeg audio extraction seems successful. Temporary audio at: ", temp_audio_path)

    -- OSD message before Python script (actual transcription)
    mp.osd_message("Parakeet: Transcribing extracted audio... This may take a while.", 7) -- MODIFIED OSD Message (kept from previous logic)
    log("info", "Step 2: Starting Parakeet transcription for: ", file_name_with_ext)

    local python_command_args = {
        python_exe,
        parakeet_script_path,
        temp_audio_path,    
        srt_output_path
    }

    log("debug", "Running Python script: ", table.concat(python_command_args, " "))
    local python_res = utils.subprocess({
        args = python_command_args,
        cancellable = false 
    })

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
            local srt_file_check = io.open(srt_output_path, "r")
            if srt_file_check then
                srt_file_check:close()
                local srt_info = utils.file_info(srt_output_path)
                if srt_info and srt_info.size > 0 then
                    mp.commandv("sub-add", srt_output_path, "auto")
                    mp.osd_message("Parakeet: Attempted to load " .. (srt_output_path:match("([^/\\]+)$") or srt_output_path), 3)
                else
                    log("warn", "SRT file found but is empty: ", srt_output_path, ". Python script might have failed to produce output.")
                    mp.osd_message("Parakeet: SRT file empty. Check console.", 5)
                end
            else
                log("warn", "SRT file not found after delay: ", srt_output_path, ". Transcription might still be running, have failed, or taken longer.")
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

log("info", "Parakeet (FFmpeg Preprocess, Full File) script loaded. Press '", key_binding, "' to transcribe.")
log("info", "Using Python from: ", python_exe)
log("info", "Using FFmpeg from: ", ffmpeg_path)
log("info", "Parakeet script (hardcoded): ", parakeet_script_path) 
log("info", "Temporary file directory: ", temp_dir)

mp.register_event("shutdown", function()
    log("info", "MPV shutdown: Checking for leftover Parakeet temporary audio files...")
end)
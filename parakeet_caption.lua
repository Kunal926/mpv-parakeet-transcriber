-- MPV integration for Parakeet ASR.
local mp = require 'mp'
local utils = require 'mp.utils'

local config = {
  python = 'python',
  parakeet = 'parakeet-transcribe',
  separator = 'parakeet-separate',
  ffmpeg = 'ffmpeg',
  ffprobe = 'ffprobe',
  temp_dir = '/tmp',
  ffmpeg_filters = 'loudnorm=I=-16:LRA=7:TP=-1.5',
  keybindings = {
    default = 'Alt+4',
    float32 = 'Alt+5',
    ffmpeg = 'Alt+6',
    ffmpeg_float32 = 'Alt+7',
    isolate_fast = 'Alt+8',
    isolate_slow = 'Alt+9',
  },
  separation = {
    fast = {
      cfg = 'weights/roformer/voc_fv4/voc_gabox.yaml',
      ckpt = 'weights/roformer/voc_fv4/voc_fv4.ckpt',
    },
    slow = {
      cfg = 'weights/roformer/karaoke_viperx/config_mel_band_roformer_karaoke.yaml',
      ckpt = 'weights/roformer/karaoke_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    },
    fp16 = false,
  },
}

local active = false
local tmp_files = {}

local function cleanup()
  for _,f in ipairs(tmp_files) do
    os.remove(f)
  end
  tmp_files = {}
end
mp.register_event('shutdown', cleanup)

local function run_subprocess(args, cb)
  mp.command_native_async({name='subprocess', args=args, playback_only=false}, function(res)
    if res.status ~= 0 then
      mp.osd_message('command failed')
      mp.msg.error('command failed: '..(res.stderr or ''))
    end
    if cb then cb(res) end
  end)
end

local function transcribe(opts)
  if active then
    mp.osd_message('transcription in progress')
    return
  end
  active = true
  local src = mp.get_property('path')
  if not src then
    mp.osd_message('no file')
    active = false
    return
  end
  local audio = opts.path or src
  local srt = utils.join_path(utils.split_path(src)) .. '.srt'
  local args = {config.parakeet, audio, '--output', srt, '--ffmpeg-path', config.ffmpeg, '--ffprobe-path', config.ffprobe}
  if opts.ffmpeg_filters then
    table.insert(args, '--ffmpeg-filters')
    table.insert(args, opts.ffmpeg_filters)
  end
  if opts.precision then
    table.insert(args, '--precision')
    table.insert(args, opts.precision)
  end
  run_subprocess(args, function()
    mp.commandv('sub-add', srt)
    active = false
  end)
end

local function isolate_and_transcribe(preset)
  local path = mp.get_property('path')
  if not path then return end
  local tmp = utils.join_path(config.temp_dir, 'vocals_'..utils.getpid()..'.wav')
  table.insert(tmp_files, tmp)
  local p = config.separation[preset]
  local args = {config.separator, path, tmp, '--cfg', p.cfg, '--ckpt', p.ckpt}
  if config.separation.fp16 then table.insert(args, '--fp16') end
  run_subprocess(args, function(res)
    if res.status == 0 then
      transcribe{ffmpeg_filters=nil, precision=nil, path=tmp}
    else
      active = false
    end
  end)
end

mp.add_key_binding(config.keybindings.default, 'parakeet_default', function() transcribe{} end)
mp.add_key_binding(config.keybindings.float32, 'parakeet_float32', function() transcribe{precision='float32'} end)
mp.add_key_binding(config.keybindings.ffmpeg, 'parakeet_ffmpeg', function() transcribe{ffmpeg_filters=config.ffmpeg_filters} end)
mp.add_key_binding(config.keybindings.ffmpeg_float32, 'parakeet_ffmpeg_float32', function() transcribe{ffmpeg_filters=config.ffmpeg_filters, precision='float32'} end)
mp.add_key_binding(config.keybindings.isolate_fast, 'parakeet_isolate_fast', function() isolate_and_transcribe('fast') end)
mp.add_key_binding(config.keybindings.isolate_slow, 'parakeet_isolate_slow', function() isolate_and_transcribe('slow') end)

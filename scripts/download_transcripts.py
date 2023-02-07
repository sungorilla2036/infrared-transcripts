from datetime import date, timedelta
import gc
import json
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr
import re
import os
import time

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import FFmpegPostProcessor,  YoutubeDL
from yt_dlp.postprocessor.common import PostProcessor
from yt_dlp.utils import (
MEDIA_EXTENSIONS,
PostProcessingError,
float_or_none,
prepend_extension,
encodeFilename
)

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '3'))

ACODECS = {
    # name: (ext, encoder, opts)
    'mp3': ('mp3', 'libmp3lame', ()),
    'aac': ('m4a', 'aac', ('-f', 'adts')),
    'm4a': ('m4a', 'aac', ('-bsf:a', 'aac_adtstoasc')),
    'opus': ('opus', 'libopus', ()),
    'vorbis': ('ogg', 'libvorbis', ()),
    'flac': ('flac', 'flac', ()),
    'alac': ('m4a', None, ('-acodec', 'alac')),
    'wav': ('wav', None, ('-f', 'wav')),
}

def get_mdx_string(content_info, video_id):
    if content_info is None:
        return f"""\
---
title: {video_id}
platform: youtube
embedId: {video_id}
---
"""
    newline = '"\n  - >\n"'
    tag_string = newline.join(content_info['tags'])
    return f"""\
---
title: >
  {content_info['title']}
date: "{content_info['upload_date']}"
platform: youtube
channelId: {content_info['channel_id']}
embedId: {content_info['id']}
views: {content_info['view_count']}
likes: {content_info['like_count']}
image:
  src: {content_info['thumbnail']}
tags:
  - >
    "{tag_string}"
---
"""

def create_mapping_re(supported):
    return re.compile(r'{0}(?:/{0})*$'.format(r'(?:\s*\w+\s*>)?\s*(?:%s)\s*' % '|'.join(supported)))


def resolve_mapping(source, mapping):
    """
    Get corresponding item from a mapping string like 'A>B/C>D/E'
    @returns    (target, error_message)
    """
    for pair in mapping.lower().split('/'):
        kv = pair.split('>', 1)
        if len(kv) == 1 or kv[0].strip() == source:
            target = kv[-1].strip()
            if target == source:
                return target, f'already is in target format {source}'
            return target, None
    return None, f'could not find a mapping for {source}'

class FFmpegPostProcessorError(PostProcessingError):
    pass

class CustomFFmpegExtractAudioPP(FFmpegPostProcessor):
    COMMON_AUDIO_EXTS = MEDIA_EXTENSIONS.common_audio + ('wma', )
    SUPPORTED_EXTS = tuple(ACODECS.keys())
    FORMAT_RE = create_mapping_re(('best', *SUPPORTED_EXTS))

    def __init__(self, downloader=None, preferredcodec=None, preferredquality=None, nopostoverwrites=False, additional_ffmpeg_args=[]):
        FFmpegPostProcessor.__init__(self, downloader)
        self.mapping = preferredcodec or 'best'
        self._preferredquality = float_or_none(preferredquality)
        self._nopostoverwrites = nopostoverwrites
        self._additional_ffmpeg_args = additional_ffmpeg_args

    def _quality_args(self, codec):
        if self._preferredquality is None:
            return []
        elif self._preferredquality > 10:
            return ['-b:a', f'{self._preferredquality}k']

        limits = {
            'libmp3lame': (10, 0),
            'libvorbis': (0, 10),
            # FFmpeg's AAC encoder does not have an upper limit for the value of -q:a.
            # Experimentally, with values over 4, bitrate changes were minimal or non-existent
            'aac': (0.1, 4),
            'libfdk_aac': (1, 5),
        }.get(codec)
        if not limits:
            return []

        q = limits[1] + (limits[0] - limits[1]) * (self._preferredquality / 10)
        if codec == 'libfdk_aac':
            return ['-vbr', f'{int(q)}']
        return ['-q:a', f'{q}']

    def run_ffmpeg(self, path, out_path, codec, more_opts):
        if codec is None:
            acodec_opts = []
        else:
            acodec_opts = ['-acodec', codec]
        opts = ['-vn'] + acodec_opts + more_opts + self._additional_ffmpeg_args
        try:
            FFmpegPostProcessor.run_ffmpeg(self, path, out_path, opts)
        except FFmpegPostProcessorError as err:
            raise PostProcessingError(f'audio conversion failed: {err.msg}')

    @PostProcessor._restrict_to(images=False)
    def run(self, information):
        orig_path = path = information['filepath']
        target_format, _skip_msg = resolve_mapping(information['ext'], self.mapping)
        if target_format == 'best' and information['ext'] in self.COMMON_AUDIO_EXTS:
            target_format, _skip_msg = None, 'the file is already in a common audio format'
        if not target_format:
            self.to_screen(f'Not converting audio {orig_path}; {_skip_msg}')
            return [], information

        filecodec = self.get_audio_codec(path)
        if filecodec is None:
            raise PostProcessingError('WARNING: unable to obtain file audio codec with ffprobe')

        if filecodec == 'aac' and target_format in ('m4a', 'best'):
            # Lossless, but in another container
            extension, _, more_opts, acodec = *ACODECS['m4a'], 'copy'
        elif target_format == 'best' or target_format == filecodec:
            # Lossless if possible
            try:
                extension, _, more_opts, acodec = *ACODECS[filecodec], 'copy'
            except KeyError:
                extension, acodec, more_opts = ACODECS['mp3']
        else:
            # We convert the audio (lossy if codec is lossy)
            extension, acodec, more_opts = ACODECS[target_format]
            if acodec == 'aac' and self._features.get('fdk'):
                acodec, more_opts = 'libfdk_aac', []

        more_opts = list(more_opts)
        if acodec != 'copy':
            more_opts = self._quality_args(acodec)

        # not os.path.splitext, since the latter does not work on unicode in all setups
        temp_path = new_path = f'{path.rpartition(".")[0]}.{extension}'

        if new_path == path:
            if acodec == 'copy':
                self.to_screen(f'Not converting audio {orig_path}; file is already in target format {target_format}')
                return [], information
            orig_path = prepend_extension(path, 'orig')
            temp_path = prepend_extension(path, 'temp')
        if (self._nopostoverwrites and os.path.exists(encodeFilename(new_path))
                and os.path.exists(encodeFilename(orig_path))):
            self.to_screen('Post-process file %s exists, skipping' % new_path)
            return [], information

        self.to_screen(f'Destination: {new_path}')
        self.run_ffmpeg(path, temp_path, acodec, more_opts)

        os.replace(path, orig_path)
        os.replace(temp_path, new_path)
        information['filepath'] = new_path
        information['ext'] = extension

        # Try to update the date time for extracted audio file.
        if information.get('filetime') is not None:
            self.try_utime(
                new_path, time.time(), information['filetime'], errnote='Cannot update utime of audio file')

        return [orig_path], information

def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-50, keep_silence=100, max_segment_len=25000):
    """
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms) amount of silence to leave at the beginning
        and end of the chunks. Keeps the sound from sounding like it is
        abruptly cut off. (default: 100ms)
    """

    not_silence_ranges = detect_nonsilent(audio_segment, min_silence_len, silence_thresh)

    chunks = []
    last_index = len(audio_segment) -1
    for start_i, end_i in not_silence_ranges:
        start_i = max(0, start_i - keep_silence)
        end_i = min(last_index, end_i + keep_silence)

        current_start_i = start_i
        current_end_i = min(current_start_i + max_segment_len, end_i)
        while current_start_i < end_i:
            chunks.append((audio_segment[current_start_i:current_end_i], current_start_i, current_end_i))
            current_start_i = current_end_i
            current_end_i = min(current_start_i + max_segment_len, end_i)

    return chunks

# For testing purposes
# print("Splitting audio...")
# audio = AudioSegment.from_wav("Hifw80gKOlM.wav")
# audio_chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-40)
# transcript = []

# chunk_files = []
# for i, chunk in enumerate(audio_chunks):
#     out_file = f"Hifw80gKOlM_{chunk[1]}_{chunk[2]}.wav"
#     chunk[0].export(out_file, format="wav")
#     chunk_files.append(out_file)

video_ids = open("video_ids.txt","r").readlines()

if (os.path.exists("errors.json")):
    with open("errors.json", "r") as f:
        errors = json.load(f)
else:
    errors = []
print(json.dumps(errors))
asr_model = None
for video_id in video_ids:
    if video_id.strip() == "" or os.path.exists(video_id.strip()+".json" or errors.count(video_id.strip()) > 0):
        print("Skipping "+video_id.strip())
        continue

    with YoutubeDL({
        'format': 'bestaudio',
        'outtmpl': video_id.strip() + '.%(ext)s'}) as ydl:
        try:
            info = ydl.extract_info(video_id.strip(), download=False)
        except:
            print("Error downloading "+video_id.strip())
            continue
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id.strip())
        except:
            try:
                if not os.path.exists(video_id.strip()+".wav"): 
                    print("No transcript for "+video_id.strip() + " downloading audio...")
                    two_days_ago = date.today() - timedelta(days=2)
                    if info["upload_date"] < two_days_ago.strftime("%Y%m%d"):
                        ydl.add_post_processor(CustomFFmpegExtractAudioPP(preferredcodec='wav', additional_ffmpeg_args=['-ac', '1', '-ar', '16000']), when='post_process')
                        ydl.download(['https://www.youtube.com/watch?v='+video_id.strip()])
                    else:
                        print("Video is too new, skipping...")
                        continue
            except:
                print("Error downloading "+video_id.strip())
                continue

            print("Splitting audio...")
            audio = AudioSegment.from_wav(video_id.strip()+".wav")
            audio_chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-40)

            chunk_files = []
            for i, chunk in enumerate(audio_chunks):
                out_file = f"{video_id.strip()}_{chunk[1]}_{chunk[2]}.wav"
                chunk[0].export(out_file, format="wav")
                chunk_files.append(out_file)
            
            #clear audio_chunks from memory
            del(audio_chunks)
            del(audio)
            gc.collect()

            transcript = []
            print("Running ASR...")
            if asr_model is None:
                asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_large")
            try:
                for fname, transcription in zip(chunk_files, asr_model.transcribe(paths2audio_files=chunk_files, batch_size=BATCH_SIZE)):
                    strings = fname.split(".")[0].split("_")
                    start = int(strings[len(strings)-2])
                    end = int(strings[len(strings)-1])
                    transcript.append({"start": start/1000, "duration": (end-start)/1000, "text": transcription})
            except:
                errors.append(video_id.strip())
            finally:
                for fname in chunk_files:
                    os.remove(fname)

        fout = open(video_id.strip() + '.json', 'x')
        json.dump(transcript, fout)
        fout.close()

        current_foldername = os.getcwd().split(os.sep)[-1]
        mdx_directory = '../../../../src/content/docs/en/youtube/' +  current_foldername + '/'
        info["upload_date"] = date(int(info["upload_date"][0:4]), int(info["upload_date"][4:6]), int(info["upload_date"][6:8])).isoformat()
        with open(mdx_directory + video_id.strip() + '.mdx', 'w') as fout:
            fout.write(get_mdx_string(info))
        print("Downloaded "+video_id.strip())
error_file = open("errors.json","w")
json.dump(errors, error_file)
error_file.close()

import subprocess
from curl_cffi import requests
import json
import os
import gc
# NeMo model
import nemo.collections.asr as nemo_asr
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

channel_info = "https://kick.com/api/v1/channels/infrared"
TRANSCRIPTS_FOLDER = "public/transcripts/kick/infrared/"

print('Getting channel info...')

response = requests.get(channel_info,  impersonate="chrome101")
response_json_obj = response.json()

streams = response_json_obj['previous_livestreams']
streams.reverse()
for stream in streams:
    uuid = stream['video']['uuid']
    if not os.path.exists(TRANSCRIPTS_FOLDER + uuid + ".json"):
        print(uuid)
        break

del(streams)
del(response_json_obj)

print('Getting stream url...')
response = requests.get("https://kick.com/api/v1/video/" + uuid,  impersonate="chrome101")
res_json = response.json()
stream_url = res_json['source']
print(stream_url)
print('Downloading stream')
subprocess.call([
    'yt-dlp',
    '-f',
    'worst',
    '-x',
    '--audio-format',
    'wav',
    '-o',
    'temp.wav',
    stream_url,
])

print('Converting to 16Khz mono wav')
subprocess.call([
    'ffmpeg',
    '-i',
    'temp.wav',
    '-ac',
    '1',
    '-ar',
    '16000',
    uuid + '.wav',
])

os.remove('temp.wav')

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

# print("Splitting audio...")
# audio = AudioSegment.from_wav(uuid + ".wav")
# audio_chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-40)

# chunk_files = []
# for i, chunk in enumerate(audio_chunks):
#     out_file = f"{uuid}_{chunk[1]}_{chunk[2]}.wav"
#     chunk[0].export(out_file, format="wav")
#     chunk_files.append(out_file)

# #clear audio_chunks from memory
# del(audio_chunks)
# del(audio)
# gc.collect()

transcript = []
print("Running ASR...")
# model_name = "nvidia/stt_en_fastconformer_transducer_xlarge"  # can also try transducer model - nvidia/stt_en_fastconformer_transducer_xlarge
# model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")

# BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))
# transcriptions = model.transcribe(paths2audio_files=chunk_files, batch_size=BATCH_SIZE)
# for fname, transcription in zip(chunk_files, transcriptions[0]):
#     strings = fname.split(".")[0].split("_")
#     start = int(strings[len(strings)-2])
#     end = int(strings[len(strings)-1])
#     transcript.append({"start": start/1000, "duration": (end-start)/1000, "text": transcription})

device = "cpu"
torch_dtype = torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=20,
    batch_size=2,
    torch_dtype=torch_dtype,
    device=device,
)

result = transcriber(uuid + ".wav", return_timestamps=True)
for chunk in result['chunks']:
    chunk['start'] = chunk['timestamp'][0]
    chunk['duration'] = chunk['timestamp'][1] - chunk['timestamp'][0]
    del(chunk['timestamp'])
transcript = result['chunks']

fout= open(TRANSCRIPTS_FOLDER + uuid + '.json', 'x')
json.dump(transcript, fout)
fout.close()

mdx_directory = 'src/content/docs/en/kick/infrared/'
with open(mdx_directory + uuid + '.mdx', 'x', encoding='utf8') as fout:
    mdx_string = f"""\
---
title: >
  {res_json['livestream']['session_title']}
date: "{res_json['livestream']['start_time']}"
platform: kick
channelId: {res_json['livestream']['channel']['slug']}
embedId: {uuid}
sourceUrl: {res_json['source']}
---
"""
    fout.write(mdx_string)
print("Downloaded "+uuid)

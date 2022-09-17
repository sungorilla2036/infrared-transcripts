import json
from os.path import exists
from youtube_transcript_api import YouTubeTranscriptApi

video_ids = open("video_ids.txt","r").readlines()

for video_id in video_ids:
    if video_id.strip() == "" or exists(video_id.strip()+".json"):
        print("Skipping "+video_id.strip())
        continue
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id.strip())
        fout = open(video_id.strip() + '.json', 'x')
        json.dump(transcript, fout)
        fout.close()
        print("Downloaded "+video_id.strip())
    except:
        #TODO run speech recognition on video to get transcript
        print(video_id)
        continue

# OpenAI project
# Video Summarizer - TL-DW (Too Long - Didn't Watch)

# Use OpenAI whisper to transcribe a youtube video into text.
# Use a prompt to tell ChatGPT to summarize it noting any key takaways.
# Use a text to voice api and play it back.

import requests
import bs4
import pytube
import whisper
import openai
import datetime
import send2trash


current_time = datetime.datetime.now().strftime("%d-%m-%Y")
# openai.api_key = "[OPEN_AI_KEY]"

#TODO - Add role assignment.
#TODO - Work on consolidating the prompts
prompt1 = "Summarize what the below youtube video transcription is about to the best of your ability and include any key take aways."
prompt2 = "Attempt a best guess at who you believe the target demographic is."
prompt3 = "Provide perspectives on the transcription that would be relevant to the target demographic."
prompt4 = "Recreate your summary for the target demographic that include your perspectives."

# TODO - replace url with user input
# Use pytube to store data from youtube video
url = "[YOUTUBE_VIDEO_LINK]" 
data = pytube.YouTube(url)

# Use pytube library to pull audio only
audio = data.streams.get_audio_only()
audio.download(filename='payload.mp4')

# Use request and bs4 library to pull the title of the video
'''result = requests.get(url)
soup = bs4.BeautifulSoup(result.text, 'lxml')
title = soup.select("title")[0].getText()'''

# Use openai whisper library to transcribe to a filename
model = whisper.load_model("base")
text = model.transcribe("payload.mp4")
transcript = text['text']

# Trash uneeded local file.
send2trash.send2trash('payload.mp4')

# OpenAI ChatGPT Prompt
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages = [{"role": "system", "content" : f"You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: {current_time}"},
{"role": "user", "content" : prompt1 + "\n\n" + transcript},
{"role": "user", "content" : prompt2},
{"role": "user", "content" : prompt3},
{"role": "user", "content" : prompt4}]
)
#print(completion)
gptSummary = completion['choices'][0]['message']['content']
print(gptSummary)

"""
URL: t.me/ukr_stt_bot
"""
import json
import warnings

warnings.simplefilter('ignore')

import os
import logging
import ffmpeg
import torch

from os import remove
from os.path import dirname

import telebot
from uuid import uuid4
from utils_vad import get_language_and_group, read_audio, init_jit_model

TOKEN = os.environ['TOKEN']

if not TOKEN:
    print('You must set the TOKEN environment variable')
    exit(1)

START_MSG = '''Hello!

This bot is created to test the language classifier built by Silero.

The repository: https://github.com/snakers4/silero-vad

The group for discussions: https://t.me/silero_speech'''

FIRST_STEP = '''It is so simple to use this bot: just send an audio message here'''

device = torch.device('cpu')

jit_model = dirname(__file__) + '/model/lang_classifier_116.jit'
model = init_jit_model(jit_model, device=device)

bot = telebot.TeleBot(TOKEN, parse_mode=None)

lang_dict_file = dirname(__file__) + '/model/lang_dict_116.json'
lang_group_dict_file = dirname(__file__) + '/model/lang_group_dict_116.json'

with open(lang_dict_file, 'r') as f:
    lang_dict = json.load(f)

with open(lang_group_dict_file, 'r') as f:
    lang_group_dict = json.load(f)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, START_MSG)
    bot.reply_to(message, FIRST_STEP)


@bot.message_handler(content_types=['voice'])
def process_voice_message(message):
    # download the recording
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # save the recording on the disk
    uuid = uuid4()
    filename = dirname(__file__) + f'/recordings/{uuid}.ogg'
    with open(filename, 'wb') as f:
        f.write(downloaded_file)

    # convert OGG to WAV
    wav_filename = dirname(__file__) + f'/recordings/{uuid}.wav'
    _, err = (
        ffmpeg
            .input(filename)
            .output(wav_filename, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=False)
    )
    if err is not None:
        bot.reply_to(message, 'Error...')
        return

    # do the classifying
    wav = read_audio(wav_filename)
    languages, language_groups = get_language_and_group(wav, model, lang_dict, lang_group_dict, top_n=2)

    languages_text = []
    for i in languages:
        text = f'Language: {i[0]} with probability: {i[-1]}'
        languages_text.append(text)
    bot.reply_to(message, '\n\n'.join(languages_text))

    language_groups_text = []
    for i in language_groups:
        text = f'Language group: {i[0]} with probability: {i[-1]}'
        language_groups_text.append(text)

    bot.reply_to(message, '\n\n'.join(language_groups_text))

    # remove the original recording
    remove(filename)

    # remove WAV file
    remove(wav_filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    bot.polling()

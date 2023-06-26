#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import os
import time
import threading
import torch
import re
import wave
from natsort import natsorted
from num2words import num2words
from transliterate import translit

def read_input():
    lines = []

    print("Введите текст для преобразования (после завершения ввода текста, введите EXIT с новой строки): ")
    while True:
        line = input()
        if line == "EXIT":
            break
        else:
            lines.append(line)

    text_raw = lines[0]
    for line in lines[1:]:
        text_raw += ' ' + line

    return text_raw

def format_and_divide_text(text_raw, max_length = 800):
    text = translit(text_raw, 'ru')
    fragments = []
    words = text.split()
    current_fragment = words[0]

    for word in words[1:]:
        if len(current_fragment) + len(word) + 1 <= max_length:
            if re.match(r'^\d+(\.)?$', word):
                word = word.replace('.', '')
                converted_word = num2words(int(word), lang='ru')
                current_fragment += ' ' + converted_word
            else:
                current_fragment += ' ' + word
        else:
            fragments.append(current_fragment)
            current_fragment = word

    fragments.append(current_fragment)
    return fragments

def text_to_wav(text):
    device = torch.device('cuda')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000
    speaker='baya'

    audio_paths = model.save_wav(text=text,
                             speaker=speaker,
                             sample_rate=sample_rate)

def move_files():
    i = 1
    while not stop_thread:
        if os.path.exists("test.wav"):
            new_filename = f"audio{i}.wav"
            os.rename("test.wav", new_filename)
            print(f"Создан {new_filename}")
            i += 1
        time.sleep(1)

def combine_wav_files():
    output_file_name = input('Введите название аудиофайла (расширение wav будет добавлено автоматически): ') + '.wav'
    directory = os.getcwd()
    files = os.listdir(directory)

    wav_files = [file for file in files if file.startswith('audio') and file.endswith('.wav')]

    pattern = r'audio(\d)+\.wav'
    wav_files = natsorted(wav_files, key=lambda x: int(x.split('audio')[1].split('.wav')[0]))

    with wave.open(output_file_name, 'wb') as combined_wav:
        for file in wav_files:
            with wave.open(os.path.join(directory, file), 'rb') as wav:
                if combined_wav.getnframes() == 0:
                    combined_wav.setparams(wav.getparams())

                audio_data = wav.readframes(wav.getnframes())

                combined_wav.writeframes(audio_data)
            os.remove(os.path.join(directory, file))

    print("Аудиофайл был создан:", output_file_name)


if __name__ == "__main__":

    text_raw = read_input()
    print('Текст был успешно прочитан скриптом.')

    stop_thread = False
    file_mover_thread = threading.Thread(target=move_files)
    file_mover_thread.start()

    fragments = format_and_divide_text(text_raw)
    print('Начинаем конвертацию текста в аудио')
    for fragment in fragments:
        text_to_wav(fragment)

    time.sleep(1)
    stop_thread = True
    file_mover_thread.join()
    print('Конвертация фрагментов окончена')

    combine_wav_files()
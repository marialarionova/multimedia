import wave
import struct
import math
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def plus_note(sample_rate, audio, freq=None, duration_milliseconds=250, volume=0.5):
    number_of_samples = duration_milliseconds * (sample_rate / 1000.0)
    if freq:
        for x in range(int(number_of_samples)):
            audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
    else:
        audio.append(0.0)


def music_file(filename, audio, sample_rate):
    wav_file = wave.open(filename, 'w')

    channels = 1
    sample_width = 2
    frames = len(audio)
    compressive_type = 'NONE'
    compressive_name = 'not compressed'

    wav_file.setparams((channels, sample_width, sample_rate, frames, compressive_type, compressive_name))

    for sample in audio:
        wav_file.writeframes(struct.pack('h', int(sample * 32767.0)))

    wav_file.close()


def plus_sample(source_audio_fname, sample_fname):
    source_audio = AudioSegment.from_wav(source_audio_fname)
    sample = AudioSegment.from_wav(sample_fname)

    new_melody = source_audio.overlay(sample)
    new_melody.export('new_melody.wav', format='wav')

    return new_melody


def fade_audio(audio):
    source_audio = AudioSegment.from_wav(audio)
    new_melody = source_audio.fade_in(3000).fade_out(3000)
    new_melody.export('new_melody.wav', format='wav')


def create_spect(audio):

    sample_rate, samples = wavfile.read(audio)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times * 1000, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [ms]')
    plt.show()


def main():
    sample_rate = 44100.

    notes_in_freq = {'C5': 523.25, 'D5': 587.33, 'E5': 659.26, 'F5': 698.46, 'G5': 783.99, 'A5': 880., 'H5': 987.77,
                     'C6': 1046.5, 'C7': 1046.5, 'D6': 1174.7,
                     'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440., 'H4': 493.88}

    audio = []

    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['A4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 500)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['E4'], 500)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['F4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['F4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['E4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['D4'], 750)
    plus_note(sample_rate, audio, duration_milliseconds=250)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['A4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 500)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['E4'], 500)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['G4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['C5'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['H4'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['D5'], 250)
    plus_note(sample_rate, audio, duration_milliseconds=100)
    plus_note(sample_rate, audio, notes_in_freq['C5'], 750)
    plus_note(sample_rate, audio, duration_milliseconds=250)

    music_file('melody.wav', audio, sample_rate)

    sample = []

    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['A5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 500)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['E5'], 500)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['F5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['F5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['E5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['D5'], 750)
    plus_note(sample_rate, sample, duration_milliseconds=250)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['A5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 500)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['E5'], 500)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['G5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['C6'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['H5'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['D6'], 250)
    plus_note(sample_rate, sample, duration_milliseconds=100)
    plus_note(sample_rate, sample, notes_in_freq['C6'], 750)
    plus_note(sample_rate, sample, duration_milliseconds=250)

    music_file('sample.wav', sample, sample_rate)

    audio, sample = 'melody.wav', 'sample.wav'
    plus_sample(audio, sample)
    audio = 'new_melody.wav'
    fade_audio(audio)

    create_spect(audio)


if __name__ == "__main__":
    main()

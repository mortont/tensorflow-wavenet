import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf
import pickle
from sklearn.feature_extraction import DictVectorizer


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def find_global_features(directory):
    '''Uses speaker-info.txt file to map speaker id to
    age, gender, and accent'''
    with open(directory + '/speaker-info.txt', 'r') as file:
        next(file)
        features = [line.split()[:4] for line in file]

    out = [{'id': speaker[0],
        'age': speaker[1],
        'gender': speaker[2],
        'accent': speaker[3],} for speaker in features]
    return out


def feature_vectorizer(feature_dict, load_file=None, save_file='vectorizer.pkl'):
    '''Vectorizes a feature dict with a saved vectorizer or
    makes its own vectorizer and saves it if one is not given.
    Returns a dense vector representation of the feature_dict'''
    if load_file:
        try:
            with open(load_file, 'r') as file:
                vectorizer = pickle.load(file)
        except:
            print('Error loading {}'.format(load_file))
    else:
        vectorizer = DictVectorizer(sparse=False).fit(feature_dict)
        with open(save_file, 'wb') as file:
            pickle.dump(vectorizer, file)

    return vectorizer.transform(feature_dict)


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

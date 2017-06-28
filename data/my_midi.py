import midi_inspect
import midi

class MyMidi(object):
    '''Abstract representation of a MIDI track. Lazily initializes different
    data representations as needed.
    '''

    def __init__(self, path=None, labels=None):
        ''' Can be initialized either with a path or with labels.
        Path is relative to AudioNN/data
        '''
        self._path = path
        self._labels = labels
        self._pattern = None
        self._note_pattern = None

    def labels(self):
        # Labels used as the output to the Neural Network
        if not self._labels:
            self._labels = midi_inspect.labelsForPath(self._path)
        return self._labels

    def pattern(self):
        # Pattern used by the midi library.
        if not self._pattern:
            if self._path:
                self._pattern = midi.read_midifile(self._path)
            elif self._labels:
                self._pattern = midi_inspect.midiFromLabels(self.labels)
        return self._pattern

    def note_pattern(self):
        # Representation of the MIDI used for audio conversion. Removes all
        # data except for notes and tempo (e.g. pedal).
        if not self._note_pattern:
            self._note_pattern = midi_inspect.noteTrackForPattern(self.pattern())
        return self._note_pattern


if __name__ == "__main__":
    my_midi = MyMidi("training_data/raw/aria.mid")
    print my_midi.note_pattern()
    print my_midi.labels()

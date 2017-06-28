import os
import subprocess
import constants
import my_midi
import midi

def _preprocess(midi_file):
    full_path = constants.AUDIO_FILE_PATH + midi_file
    mmidi = my_midi.MyMidi(path=full_path)
    tmp_write_path = "/tmp/tmp_midi.mid"
    midi.write_midifile(tmp_write_path, mmidi.note_pattern())
    return tmp_write_path

def _process(base_name, midi_pattern):
    mp3_name = base_name + '.mp3'
    print 'Creating: ' + mp3_name
    subprocess.check_call("timidity '%s' -Ow -o - | lame - -b 64 '%s'" %
            (preprocessed_midi_path,
             constants.MIDI_FILE_PATH + mp3_name), shell=True)
def main():
  """ Creates a set of audio files given a list of midi files. Requires timidity and lame."""
  for midi_name in os.listdir(constants.TRAINING_DATA_PATH),:
    print 'Processing: ' + midi_name
    base_name = os.path.splitext(midi_name)[0]
    preprocessed_midi_path = _preprocess(midi_name)
    process(base_name, midi.read_midifile(preprocessed_midi_path))

if __name__ == "__main__":
    #main()
    midi_name = "chpn_op25_e12.mid"
    preprocessed_midi_path = _preprocess(midi_name)

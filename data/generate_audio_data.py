import os
import subprocess
import constants
import my_midi
import midi

def _process(base_name, midi_path):
    mp3_name = base_name + '.mp3'
    print 'Creating: ' + mp3_name
    subprocess.check_call("timidity '%s' -Ow -o - | lame - -b 64 '%s'" %
            (midi_path,
             constants.AUDIO_FILE_PATH + mp3_name), shell=True)
def main():
  """ Creates a set of audio files given a list of midi files. Requires timidity and lame."""
  for midi_name in os.listdir(constants.MIDI_FILE_PATH):
    if not midi_name.endswith(".mid"):
        continue
    print 'Processing: ' + midi_name
    base_name = os.path.splitext(midi_name)[0]
    _process(base_name, midi_name)

if __name__ == "__main__":
    main()

import os
import subprocess
import constants

def main():
  """ Creates a set of audio files given a list of midi files. Requires timidity and lame."""
  for midi_name in os.listdir(constants.TRAINING_DATA_PATH),:
    print 'Processing: ' + midi_name
    base_name = os.path.splitext(midi_name)[0]
    mp3_name = base_name + '.mp3'
    print 'Creating: ' + mp3_name
    subprocess.check_call("timidity '%s' -Ow -o - | lame - -b 64 '%s'" %
            (constants.AUDIO_FILE_PATH + midi_name,
             constants.MIDI_FILE_PATH + mp3_name), shell=True)

if __name__ == "__main__":
  main()

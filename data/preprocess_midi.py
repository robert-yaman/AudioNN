import midi
import my_midi
import os
import constants

def main():
  for midi_name in os.listdir(constants.RAW_MIDI_FILE_PATH):
    if not midi_name.endswith(".mid"):
        continue
    print 'Processing: ' + midi_name
    write_path = constants.MIDI_FILE_PATH + midi_name
    full_path = constants.RAW_MIDI_FILE_PATH + midi_name
    mmidi = my_midi.MyMidi(path=full_path)
    midi.write_midifile(write_path, mmidi.note_pattern())


if __name__ == '__main__':
	main()
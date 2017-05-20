import midi
import sys
import os
import copy
import argparse 

# currently mutates
# There's a bug with tempo changes I think, but We actually don't have to do this - now we can just maintain concurrent
# iterations.
    # Actually there isn't a bug, it was just shit midi
def patternWithOneNoteTrackFromPattern(pattern):
  # Takes a MIDI track with multiple note tracks (e.g. for a Bach fugue) and combines them. Returns a pattern with the unified track

  note_tracks = []
  for track in pattern:
    if type(track[0]) == midi.TextMetaEvent:
      note_tracks.append(track)

  final_track = midi.Track()
  meta = midi.TextMetaEvent(tick=0, text='final', data=[49])
  final_track.append(meta)

  track_abs_times = {}
  for track in note_tracks:
      track_abs_times[track[0].text] = 0 #TODO:confirm these are always unique 
  total_time = 0
  # get rid of the meta event tag in each tr ack
  # close - also end of track event
  while any([len(track) > 1 for track in note_tracks]):
    track_with_next_event = None
    best_score = float('inf') # we want low scores
    for itr_track in note_tracks:
      if len(itr_track) > 1:
        if track_with_next_event == None or track_abs_times[itr_track[0].text] + itr_track[1].tick < best_score:
          track_with_next_event = itr_track
          best_score = track_abs_times[itr_track[0].text] + itr_track[1].tick
    # 0'th index is the meta event
    next_event = track_with_next_event.pop(1) # this would be more efficient if we went backwards

    track_abs_times[track_with_next_event[0].text] += next_event.tick
    if type(next_event) == midi.NoteOnEvent or type(next_event) == midi.NoteOffEvent:
      next_event.tick = track_abs_times[track_with_next_event[0].text] - total_time
      final_track.append(next_event)
    total_time = track_abs_times[track_with_next_event[0].text]

  # Preserve metadata on track and tempo track.
  while len(pattern) > 1:
    pattern.pop()

  pattern.append(final_track)
  return pattern

def _string_from_note(note):
    note_value_dict = {
            0 : "A",
            1 : "A#",
            2 : "B",
            3 : "C",
            4 : "C#",
            5 : "D",
            6 : "D#",
            7 : "E",
            8 : "F",
            9 : "F#",
            10 : "G",
            11 : "G#"
    }
    # Add three to the note since C is 0 in MIDI but A is the lowest note on a
    # piano
    note += 3
    note_string = note_value_dict[note % 12]
    note_octave = note / 12
    return note_string + str(note_octave)

def _display_one_hot(list):
    print "Notes: "
    note_string = ""
    index = 0
    for note in list:
        if note:
            note_string += _string_from_note(index) + ", "
        index += 1
    print note_string

def _microseconds_per_tick(resolution, mpqn):
    # |mpqn| microseconds per beat
    # |resolution| ticks per beat
    return mpqn / resolution

# Two problems:
    # not enough data points at the end
    # missing the first note
def labelsForTrack(pattern, interval=23217):
    # Returns 2d numpy array of shape (88, N). Each slice along dim 2 is a one
    # hot encoding of the notes currently sounding at slice n of the pattern.
    # |interval| is in terms of microseconds, and is invariate to tempo changes.
    #
    # Assumption: the first track of |pattern| contains only tempo changes. The
    # second track of pattern contains only noteOn and noteOff events.
    # (remember that ticks change duration base on the tempo. Interestingly it
    # seems like one note can span ticks that have different values if it
    # stradles a tempo change!)
    pattern.make_ticks_abs()

    answer  = []
    time_per_tick = 0
    # Use a one-hot encoding for currently sounding notes, since this is what
    # we will use as output representation.
    current_notes = [0] * 88
    current_time_mod = 0

    tempo_track = pattern[0]
    note_track = pattern[1]
    tempo_index = 0
    note_index = 0

    # Used to keep track of how far each event is from the previous event.
    last_tick_processed = 0
    while tempo_index < len(tempo_track) and note_index < len(note_track):
        if current_time_mod > interval:
            answer.append(current_notes[:])
            # _display_one_hot(current_notes)
            current_time_mod = current_time_mod - interval
        # Figure out whats the next event to process
        next_tempo_event = tempo_track[tempo_index]
        next_note_event = note_track[note_index]
        if next_tempo_event.tick <= next_note_event.tick:
            if not type(next_tempo_event) == midi.SetTempoEvent:
                tempo_index += 1
                continue
            tick_change = next_tempo_event.tick - last_tick_processed
            last_tick_processed = next_tempo_event.tick
            # BUG - we're jumping over a bunch of MFCCs here.
                # need an extra step where if the next step is past where the
                # next MFCC starts, we jump to that spot, log the MFCC and try
                # again
            current_time_mod += tick_change * time_per_tick

            tempo_index += 1
            time_per_tick = _microseconds_per_tick(pattern.resolution,
                    next_tempo_event.mpqn)
            print time_per_tick
        else:
            if not (type(next_note_event) == midi.NoteOnEvent or
                    type(next_note_event) == midi.NoteOffEvent):
                note_index += 1
                continue
            tick_change = next_note_event.tick - last_tick_processed
            last_tick_processed = next_note_event.tick

            current_time_mod += tick_change * time_per_tick

            note_index += 1
            pitch = next_note_event.get_pitch()
            if type(next_note_event) == midi.NoteOnEvent:
                if current_notes[pitch]:
                    print("ERROR: Note already ON: %d" % pitch)
                current_notes[pitch] = 1
            else:
                if not current_notes[pitch]:
                    print("ERROR: Note already OFF: %d" % pitch)
                current_notes[pitch] = 0
    return answer


# Note that tempo events do actually affect audio
def removeTempoEvents(pattern):
    tempo_track = pattern[0]
    # Take the initial time signature, the initial tempo, and the end of
    # track.
    new_tempo_track = tempo_track[0:2] + [tempo_track[-1]]
    pattern[0] = new_tempo_track
    return pattern

if __name__ == "__main__":
    midi_path = os.getcwd() + '/' + sys.argv[1]
    pattern =midi.read_midifile(midi_path)
    print(len(labelsForTrack(pattern)))

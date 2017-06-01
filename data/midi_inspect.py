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

class _MFCCEvent(object):
    ''' Represents the beginning of a new MFCC window.'''
    def __init__(self, time):
        self.time = time

# This yields fewer data points than the MFCC processing, I think because this
# doesn't take into account terminating silence. Therefore, just discard the
# leftover MFCC pieces during training.
def labelsForTrack(pattern, interval=23217):
    # Returns 2d numpy array of shape (88, N). Each slice along dim 2 is a one
    # hot encoding of the notes currently sounding at slice n of the pattern.
    # |interval| is in terms of microseconds, and is invariate to tempo changes.
    #
    # Assumption: the first track of |pattern| contains only tempo changes. The
    # second track of pattern contains only noteOn and noteOff events.
    pattern.make_ticks_abs()

    answer  = []
    time_per_tick = 0
    # Use a one-hot encoding for currently sounding notes, since this is what
    # we will use as output representation.
    current_notes = [0] * 88
    # Have to keep track of both time and tick value since MFFCs are in terms
    # of microseconds, and MIDI events are in terms of ticks, and ticks can
    # change micrsecond value based on tempo change events.
    current_time = 0
    last_tick_processed = 0
    last_processed_midi_event_time = 0

    tempo_track = pattern[0]
    note_track = pattern[1]
    tempo_index = 0
    note_index = 0

    while tempo_index < len(tempo_track) and note_index < len(note_track):
        # find event (note, tempo, mfcc), process
        next_tempo_event = tempo_track[tempo_index]
        next_note_event = note_track[note_index]
        next_mfcc_event =  _MFCCEvent((current_time - (current_time % interval)) + interval)

        next_tempo_time = (last_processed_midi_event_time +
                (next_tempo_event.tick - last_tick_processed) *
                time_per_tick)
        next_note_time = (last_processed_midi_event_time +
                (next_note_event.tick - last_tick_processed) *
                time_per_tick)
        next_mfcc_time = next_mfcc_event.time

        min_time = min(next_tempo_time, next_note_time, next_mfcc_time)

        if next_tempo_time == min_time:
            # Process tempo event
            # Process tempos first so that the initial tempo is recorded.
            tempo_index += 1
            if not type(next_tempo_event) == midi.SetTempoEvent:
                continue
            time_per_tick = _microseconds_per_tick(pattern.resolution,
                next_tempo_event.mpqn)

            last_tick_processed = next_tempo_event.tick
            current_time = next_tempo_time
            last_processed_midi_event_time = next_tempo_time
        elif next_note_time == min_time:
            # Process note event
            note_index += 1
            if not (type(next_note_event) == midi.NoteOnEvent or
                    type(next_note_event) == midi.NoteOffEvent):
                continue

            pitch = next_note_event.get_pitch()
            if type(next_note_event) == midi.NoteOnEvent:
                if current_notes[pitch]:
                    print("ERROR: Note already ON: %d" % pitch)
                current_notes[pitch] = 1
            else:
                if not current_notes[pitch]:
                    print("ERROR: Note already OFF: %d" % pitch)
                current_notes[pitch] = 0

            last_tick_processed = next_note_event.tick
            current_time = next_note_time
            last_processed_midi_event_time = next_note_time
        elif next_mfcc_time == min_time:
            # Process MFCC event
            answer.append(current_notes[:])
            # print _display_one_hot(current_notes)
            current_time = next_mfcc_time
        else:
            print "ERROR: Min time is miscalculated."
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
import midi
import sys
import os
import copy

# currently mutates
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

# Actually, we should just go straight into the representation we're using for our model here. Otherwise, the MIDI won't necessarily be valid (e.g. if there is a holdover note between measures).
# Might be useful to be able to define ALL events into a single track for this.
def DividePatternIntoMeasure(pattern):
  # returns list of patterns 

if __name__ == "__main__":
  midi_path = os.getcwd() + '/' + sys.argv[1]
  pattern =midi.read_midifile(midi_path)
  midi.write_midifile("OneTrackFugue.midi",patternWithOneNoteTrackFromPattern(pattern))


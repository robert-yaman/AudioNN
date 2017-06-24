import midi
import sys
import os
import copy
import argparse

# Difference between the MIDI value of a note and the piano value (lowest A is
# 0).
MIDI_OFFSET = 21

def noteTrackForPattern(pattern):
    pattern.make_ticks_abs()

    # May also include pedal and metadata tracks, but we'll ignore those for
    # now.
    note_tracks = pattern[1:]
    final_track = midi.Track()
    meta = midi.TextMetaEvent(tick=0, text='final', data=[49])
    final_track.append(meta)
    while any([len(x) > 0 for x in note_tracks]):
        track_with_next_event = note_tracks[0]
        next_event_tick = float("inf")
        for note_track in note_tracks:
            if len(note_track) > 0 and note_track[0].tick < next_event_tick:
                track_with_next_event = note_track
                next_event_tick = track_with_next_event[0].tick
        next_event = track_with_next_event.pop(0)
        if (type(next_event) == midi.NoteOnEvent or type(next_event) ==
                midi.NoteOffEvent):
            final_track.append(next_event)
    final_track.append(midi.EndOfTrackEvent(tick=0, data=[]))

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
    return mpqn * 1.0 / resolution

class _MFCCEvent(object):
    ''' Represents the beginning of a new MFCC window.'''
    def __init__(self, time):
        self.time = time

# This yields fewer data points than the MFCC processing, I think because this
# doesn't take into account terminating silence. Therefore, just discard the
# leftover MFCC pieces during training.
def labelsForNoteTrack(pattern, interval=23217, verbose=False):
    # Returns 2d numpy array of shape (88, N). Each slice along dim 2 is a one
    # hot encoding of the notes currently sounding at slice n of the pattern.
    # |interval| is in terms of microseconds, and is invariate to tempo changes.
    #
    # Assumption: the first track of |pattern| contains only tempo changes. The
    # second track of pattern contains only noteOn and noteOff events.

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

    while tempo_index < len(tempo_track) or note_index < len(note_track):
        if tempo_index < len(tempo_track):
            next_tempo_event = tempo_track[tempo_index]
            next_tempo_time = (last_processed_midi_event_time +
                    (next_tempo_event.tick - last_tick_processed) *
                    time_per_tick)
        else:
            next_tempo_event = None
            next_tempo_time = float("inf")
        if note_index < len(note_track):
            next_note_event = note_track[note_index]
            next_note_time = (last_processed_midi_event_time +
                    (next_note_event.tick - last_tick_processed) *
                    time_per_tick)
        else:
            next_note_event = None
            next_note_time = float("inf")

        next_mfcc_event =  _MFCCEvent((current_time - (current_time % interval)) + interval)
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

            pitch = next_note_event.get_pitch() - MIDI_OFFSET
            if type(next_note_event) == midi.NoteOnEvent:
                # Sometimes two NoteOnEvents represent an On/Off pair.
                current_notes[pitch] = 1 - current_notes[pitch]
            elif type(next_note_event) == midi.NoteOffEvent:
                if not current_notes[pitch]:
                    print("ERROR: Note already OFF: %d" % pitch)
                current_notes[pitch] = 0

            last_tick_processed = next_note_event.tick
            current_time = next_note_time
            last_processed_midi_event_time = next_note_time
        elif next_mfcc_time == min_time:
            # Process MFCC event
            answer.append(current_notes[:])
            if verbose:
               _display_one_hot(current_notes)
            current_time = next_mfcc_time
        else:
            print "ERROR: Min time is miscalculated."
    return answer


# Note that tempo events do actually affect audio
def removeTempoEvents(pattern):
    tempo_track = pattern[0]
    # Take the initial time signature, the initial tempo, an the end of
    # track.
    new_tempo_track = tempo_track[0:2] + [tempo_track[-1]]
    pattern[0] = new_tempo_track
    return pattern

def labelsForPath(path, verbose=False):
    pattern = midi.read_midifile(path)
    return labelsForNoteTrack(noteTrackForPattern(pattern), verbose=verbose)

def midiFromLabels(labels, interval=23217):
    """Method to take output of our system and create a MIDI file we can put
    through a synthesizer. |labels| is a list of 1x88 matricies of one hot
    encodings of possible notes. Intervals is the number of microseconds
    between each encoding.

    returns Midi track. For now tempo is 60 bpm in all cases.
    """
    # Default resolution 220.
    pattern = midi.Pattern(tick_relative=False)
    tempo_track = midi.Track(tick_relative=False)
    tempo_event = midi.SetTempoEvent(tick=0, bpm=60)
    tempo_track.append(tempo_event)
    pattern.append(tempo_track)
    # Don't create a tempo track for now - use all defaults.
    note_track = midi.Track(tick_relative=False)

    pattern.append(note_track)
    # Hash set
    currently_on_notes = {}

    def recordChangeAtIndex(index, tick):
        midi_index = index + MIDI_OFFSET
        if currently_on_notes.get(index, False):
            event = midi.NoteOffEvent
        else:
            event = midi.NoteOnEvent
        note_track.append(event(tick=tick, pitch=midi_index, velocity=64))
        currently_on_notes[index] = not currently_on_notes.get(index,
            False)

    def tickFromTime(time):
        # |time| in microseconds.
        ticks_per_sec = 220.0
        ticks_per_microsecs = ticks_per_sec / 1000000
        return int(ticks_per_microsecs * time)

    current_time = 0
    for label in labels:
        current_time += interval
        for index, value in enumerate(label):
            if currently_on_notes.get(index, False) != value:
                recordChangeAtIndex(index, tickFromTime(current_time))

    final_tick = note_track[-1].tick
    note_track.append(midi.EndOfTrackEvent(tick=final_tick))
    tempo_track.append(midi.EndOfTrackEvent(tick=final_tick))

    pattern.make_ticks_rel()
    return pattern

if __name__ == "__main__":
    midi_path = sys.argv[1]
    pattern = midiFromLabels(labelsForPath(midi_path))
    print pattern
    midi.write_midifile("/tmp/test_midi.mid", pattern)
    print "Done"

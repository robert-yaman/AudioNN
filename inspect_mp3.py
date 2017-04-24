import mad, ao, sys

mf = mad.MadFile(sys.argv[1])
dev = ao.AudioDevice('oss', rate=mf.samplerate())
while 1:
    buf = mf.read()
    if buf is None:
        break
    dev.play(buf, len(buf))


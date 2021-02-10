#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# this file came from: https://github.com/lamaun/jumpcutter/

import subprocess
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import rmtree, move, copyfile
import os
import argparse
from pytube import YouTube
from time import time
import distutils.util
#import tempfile

#def mywav_write_header(fid):
#    fs = rate

#    dkind = data.dtype.kind
#    if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and
#                                                data.dtype.itemsize == 1)):
#        raise ValueError("Unsupported data type '%s'" % data.dtype)

#    header_data = b''

#    header_data += b'RIFF'
#    header_data += b'\x00\x00\x00\x00'
#    header_data += b'WAVE'

#    # fmt chunk
#    header_data += b'fmt '
#    if dkind == 'f':
#        format_tag = WAVE_FORMAT_IEEE_FLOAT
#    else:
#        format_tag = WAVE_FORMAT_PCM
#    if data.ndim == 1:
#        channels = 1
#    else:
#        channels = data.shape[1]
#    bit_depth = data.dtype.itemsize * 8
#    bytes_per_second = fs*(bit_depth // 8)*channels
#    block_align = channels * (bit_depth // 8)

#    fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs,
#                                    bytes_per_second, block_align, bit_depth)
#    if not (dkind == 'i' or dkind == 'u'):
#        # add cbSize field for non-PCM files
#        fmt_chunk_data += b'\x00\x00'

#    header_data += struct.pack('<I', len(fmt_chunk_data))
#    header_data += fmt_chunk_data

#    # fact chunk (non-PCM files)
#    if not (dkind == 'i' or dkind == 'u'):
#        header_data += b'fact'
#        header_data += struct.pack('<II', 4, data.shape[0])

#    # check data size (needs to be immediately before the data chunk)
#    if ((len(header_data)-4-4) + (4+4+data.nbytes)) > 0xFFFFFFFF:
#        raise ValueError("Data exceeds wave file size limit")

#    fid.write(header_data)


#def mywav_update_header(filename):
#    fid = open(filename, 'wb')
#    fid.seek(0)
#    mywav_update_header(fid)
#    fid.truncate()
#    fid.close()


#def mywav_write_file(filename, rate, data):
#    fid = open(filename, 'wb')

#    mywav_write_header(fid)

#    # data chunk
#    fid.write(b'data')
#    fid.write(struct.pack('<I', data.nbytes))
#    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
#        data = data.byteswap()
#    _array_tofile(fid, data)

#    # Determine file size and place it in correct
#    #  position at start of the file.
#    size = fid.tell()
#    fid.seek(4)
#    fid.write(struct.pack('<I', size-8))

#    fid.close()



def safe_remove(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def downloadFile(url):
    sep = os.path.sep
    originalPath = YouTube(url).streams.first().download()
    filepath = originalPath.split(sep)
    filepath[-1] = filepath[-1].replace(' ','_')
    filepath = sep.join(filepath)
    os.rename(originalPath, filepath)
    return filepath


def getFrameRate(path):
    process = subprocess.Popen(["ffmpeg", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    output =  stdout.decode()
    match_dict = re.search(r"\s(?P<fps>[\d\.]+?)\stbr", output).groupdict()
    return float(match_dict["fps"])

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(inputFrame,outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    #copyfile(src, dst)
    os.rename(src, dst)
    ## Remove unneeded frames
    #inputFrame-=1
    #src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    #while safe_remove(src):
	   # inputFrame-=1
	   # src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def deletePathAndExit(s, msg="", rc=0): # Dangerous! Watch out!
    rmtree(s)
    print(msg)
    exit(rc)

def writeELD(start, end, number):
    startFrame = int(start % frameRate)
    startSecond = int((start / frameRate) % 60)
    startMinute = int((start / frameRate / 60) % 60)
    startHour = int((start / frameRate / 60 / 60))
    endFrame = int(end % frameRate)
    endSecond = int((end / frameRate) % 60)
    endMinute = int((end / frameRate / 60) % 60)
    endHour = int((end / frameRate / 60 / 60))
    eld_file = open(OUTPUT_FILE, "a")
    eld_file.write("{0} 001 V C {4}:{3}:{2}:{1} {8}:{7}:{6}:{5} {4}:{3}:{2}:{1} {8}:{7}:{6}:{5}\r\n".format(
        str(number).zfill(3),
        str(startFrame).zfill(2),
        str(startSecond).zfill(2),
        str(startMinute).zfill(2),
        str(startHour).zfill(2),
        str(endFrame).zfill(2),
        str(endSecond).zfill(2),
        str(endMinute).zfill(2),
        str(endHour).zfill(2)
    ))
    eld_file.close()

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('-i', '--input_file', type=str,  help='the video file you want modified')
parser.add_argument('-u', '--url', type=str, help='A youtube url to download and process')
parser.add_argument('-o', '--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
parser.add_argument('-f', '--force', default=False, action='store_true', help='Overwrite output_file without asking')
parser.add_argument('-t', '--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('-snd', '--sounded_speed', type=float, default=1.0, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('-sil', '--silent_speed', type=float, default=999999.0, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('-fm', '--frame_margin', type=float, default=0, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
parser.add_argument('-sr', '--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('-fr', '--frame_rate', type=float, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('-fq', '--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")
parser.add_argument('-p', '--preset', type=str, default="medium", help="A preset is a collection of options that will provide a certain encoding speed to compression ratio. See https://trac.ffmpeg.org/wiki/Encode/H.264")
parser.add_argument('-crf', '--crf', type=int, default=23, help="Constant Rate Factor (CRF). Lower value - better quality but large filesize. See https://trac.ffmpeg.org/wiki/Encode/H.264")
parser.add_argument('-alg', '--stretch_algorithm', type=str, default="wsola", help="Sound stretching algorithm. 'phasevocoder' is best in general, but sounds phasy. 'wsola' may have a bit of wobble, but sounds better in many cases.")
parser.add_argument('-a', '--audio_only', default=False, action='store_true', help="outputs an audio file")
parser.add_argument('-edl', '--edl', default=False, action='store_true', help='EDL export option. (Supports only cuts off)')

try: # If you want bash completion take a look at https://pypi.org/project/argcomplete/
    import argcomplete
    argcomplete.autocomplete(parser)
except ImportError:
    pass
args = parser.parse_args()



frameRate = args.frame_rate
SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
AUDIO_ONLY = args.audio_only
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url != None:
    INPUT_FILE = downloadFile(args.url)
else:
    INPUT_FILE = args.input_file
URL = args.url
FRAME_QUALITY = args.frame_quality
EDL = args.edl
FORCE = args.force
H264_PRESET = args.preset
H264_CRF = args.crf

STRETCH_ALGORITHM = args.stretch_algorithm
if(STRETCH_ALGORITHM == "phasevocoder"):
    from audiotsm import phasevocoder as audio_stretch_algorithm
elif (STRETCH_ALGORITHM == "wsola"):
    from audiotsm import wsola as audio_stretch_algorithm
else:
    raise Exception("Unknown audio stretching algorithm.")

assert INPUT_FILE != None , "why u put no input file, that dum"
assert os.path.isfile(INPUT_FILE), "I can't read/find your input file"
assert FRAME_QUALITY < 32 , "The max value for frame quality is 31."
assert FRAME_QUALITY > 0 , "The min value for frame quality is 1."

if len(args.output_file) >= 1:
    OUTPUT_FILE = args.output_file
else:
    OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

if FORCE:
    safe_remove(OUTPUT_FILE)
else:
    if os.path.isfile(OUTPUT_FILE):
        if distutils.util.strtobool(input(f"Do you want to overwrite {OUTPUT_FILE}? (y/n)")):
            safe_remove(OUTPUT_FILE)
        else:
            exit(0)

TEMP_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TEMP")
if os.path.exists(TEMP_FOLDER):
    rmtree(TEMP_FOLDER)
os.mkdir(TEMP_FOLDER)

AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

if not (AUDIO_ONLY or EDL):
    command = ["ffmpeg", "-i", INPUT_FILE, "-qscale:v", str(FRAME_QUALITY), TEMP_FOLDER+"/frame%06d.jpg", "-hide_banner"]
    rc = subprocess.run(command)
    if rc.returncode != 0:
        deletePathAndExit(TEMP_FOLDER,"The input file doesn't have any video. Try --audio_only",rc.returncode)

command = ["ffmpeg", "-i", INPUT_FILE, "-ab", "160k", "-ac", "2", "-ar", str(SAMPLE_RATE), "-vn" ,TEMP_FOLDER+"/audio.wav"]
rc = subprocess.run(command)
if rc.returncode != 0:
    deletePathAndExit(TEMP_FOLDER,"The input file doesn't have any sound.",rc.returncode)

sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
audioSampleCount = audioData.shape[0]
maxAudioVolume = getMaxVolume(audioData)

if frameRate is None:
    try:
        frameRate = getFrameRate(INPUT_FILE)
    except AttributeError:
        if AUDIO_ONLY:
            frameRate = 1
        else:
            deletePathAndExit(TEMP_FOLDER,"Couldn't detect a framerate.",rc.returncode)

samplesPerFrame = sampleRate/frameRate

audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

hasLoudAudio = np.zeros((audioFrameCount)) # TODO: force byte dtype to save memory



for i in range(audioFrameCount):
    start = int(i*samplesPerFrame)
    end = min(int((i+1)*samplesPerFrame),audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
    if maxchunksVolume >= SILENT_THRESHOLD:
        hasLoudAudio[i] = 1

chunks = [[0,0,0]] # [startAudioFrame, endAudioFrame, shouldIncludeChunk?]
shouldIncludeFrame = np.zeros((audioFrameCount)) # TODO: no need for an array, only need to keep track of previous and current flag
for i in range(audioFrameCount):
    start = int(min(max(0,i-FRAME_SPREADAGE),audioFrameCount))
    end = int(max(0,min(audioFrameCount,i+1+FRAME_SPREADAGE)))
    if(start>end):
        end=start+1
        if(end>audioFrameCount):
            continue
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end]) # TODO: change max to find(1)
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
        chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]]) # create new chunk starting from the end of previous and finishing at current i value

chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
chunks = chunks[1:]
outputAudioData = np.empty(audioData.shape)
outputPointer = 0

mask = [x/AUDIO_FADE_ENVELOPE_SIZE for x in range(AUDIO_FADE_ENVELOPE_SIZE)] # Create audio envelope mask

lastExistingFrame = None
if EDL:
    edlFrameNumber = 0

for chunk in chunks:
    if not chunk[2]:
        continue

    if EDL:
        if (chunk[2] == True):
            edlFrameNumber += 1
            writeELD(chunk[0], chunk[1], edlFrameNumber)
        continue
    noiseChunk = np.divide(audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)], maxAudioVolume)
    
    # TODO: to improve speed, skip this entire block. we will not have support for speeding up the video.
    # TODO: all we are gonna do is throw away silent audio chunks.
    #sFile = TEMP_FOLDER+"/tempStart.wav"
    #eFile = TEMP_FOLDER+"/tempEnd.wav"
    #wavfile.write(sFile,SAMPLE_RATE,audioChunk)
    #with WavReader(sFile) as reader:
    #    with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
    #        tsm = audio_stretch_algorithm(reader.channels, speed=NEW_SPEED[int(chunk[2])])
    #        tsm.run(reader, writer)
    #_, alteredAudioData = wavfile.read(eFile)
    leng = noiseChunk.shape[0]
    endPointer = outputPointer+leng
    
    #outputAudioData.extend((audioChunk/maxAudioVolume).tolist()) # TODO: pre-allocate output audio array as np.array with original size (new size will always be smaller than or equal to old)

    # Smoothing the audio
    if noiseChunk.shape[0] < AUDIO_FADE_ENVELOPE_SIZE:
        for i in range(noiseChunk.shape[0]):
            noiseChunk[i][0] = 0
            noiseChunk[i][1] = 0
    else:
        for i in range(0, AUDIO_FADE_ENVELOPE_SIZE):
            noiseChunk[i][0] *= mask[i]
            noiseChunk[i][1] *= mask[i]
        for i in range(noiseChunk.shape[0]-AUDIO_FADE_ENVELOPE_SIZE, noiseChunk.shape[0]):
            noiseChunk[i][0] *= (1-mask[i-noiseChunk.shape[0]+AUDIO_FADE_ENVELOPE_SIZE])
            noiseChunk[i][1] *= (1-mask[i-noiseChunk.shape[0]+AUDIO_FADE_ENVELOPE_SIZE])
    if not AUDIO_ONLY:
        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(inputFrame,outputFrame) # TODO: instead of copying can't we just rename it??
            if outputFrame % 1000 == 0:
                print(str(inputFrame + 1) + "/" + str(audioFrameCount) + " frames processed.", end="\r", flush=True)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(lastExistingFrame,outputFrame) # TODO: instead of copying can't we just rename it??
    outputAudioData[outputPointer:endPointer] = noiseChunk
    outputPointer = endPointer

wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData[0:outputPointer]) # TODO: get write() func src and write incrementally
#outputAudioData =  np.asarray(outputAudioData) # TODO: no need for this
#if not EDL:
#    wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

'''
outputFrame = math.ceil(outputPointer/samplesPerFrame)
for endGap in range(outputFrame,audioFrameCount):
    copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
'''
if not EDL:
    if AUDIO_ONLY:
        command = ["ffmpeg", "-i", TEMP_FOLDER+"/audioNew.wav", OUTPUT_FILE]
    else:
        command = ["ffmpeg", "-framerate", str(frameRate), "-i", TEMP_FOLDER+"/newFrame%06d.jpg", "-i", TEMP_FOLDER +
               "/audioNew.wav", "-strict", "-2", "-c:v", "libx264", "-preset", str(H264_PRESET), "-crf", str(H264_CRF), "-pix_fmt", "yuvj420p", OUTPUT_FILE]
    rc = subprocess.run(command)
    if rc.returncode != 0:
        deletePathAndExit(TEMP_FOLDER,rc,rc.returncode)

deletePathAndExit(TEMP_FOLDER)

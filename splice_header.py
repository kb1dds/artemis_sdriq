import numpy as np

with open('/home/michaelr/20260402_ArtemisII/camras-2022_11_30_19_17_04_2216.500MHz_5.0Msps_ci16_le.sigmf-data', 'rb') as fp:
    data = np.fromfile(fp, dtype='int16', sep = '')

sample_rate = np.array(5e6,dtype='uint32')
center_freq = np.array(2.2165e9, dtype='uint64')
timestamp = np.array(0,dtype='uint64')
sample_size = np.array(16,dtype='uint32')
crc = np.array(0,dtype='uint32')

with open('artemis1_camras.sdriq','wb') as fp:
    fp.write(sample_rate.tobytes())
    fp.write(center_freq.tobytes())
    fp.write(timestamp.tobytes())
    fp.write(sample_size.tobytes())
    fp.write(crc.tobytes()) # Padding
    fp.write(crc.tobytes())
    fp.write(data.tobytes())

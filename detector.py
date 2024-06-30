from collections import namedtuple
from os import environ, path
from sys import argv

import numpy as np
import wfdb
from scipy.signal import medfilt
from scipy.stats import trim_mean
from wfdb.processing import Comparitor, resample_sig

# Algorithm parameters
F_SAMPLE = 250
ALPHA = 0.25
W_ECG = 0.2
W_BP = 1.0
L_ECG = 0.2
L_BP = 0.4

Record = namedtuple('Record', 'record annotations')
Detections = namedtuple('Detections', 'channel type beats noisy_parts')

dataset_dir = ''
BP_STRINGS = ['pressure', 'bp', 'cvp', 'pa', 'pap', 'art']


def main():
    if len(argv) != 2:
        print(
            'Assignment 1: QRS detection\n'
            'Program detects QRS complexes in the given record using multiple\n'
            'channels. Annotations are written to the .asc file in the current\n'
            'directory. Evaluation using WFDB is outputted to the console.\n\n'
            'Usage: python detector.py <record>\n'
            'Examples:\n'
            '\tpython detector.py 100\n'
            '\tpython detector.py ../databases/set-p/100'
        )
        exit(1)

    record = argv[1]

    global dataset_dir
    dataset_dir = path.dirname(record)
    record = path.basename(record)

    eval_record(record, verbose=True)


def read_record(record):
    if not globals()['dataset_dir']:
        dataset_dir = environ.get('DATASET_DIR', '')

    signals = wfdb.rdrecord(path.join(dataset_dir, record))
    annotations = wfdb.rdann(path.join(dataset_dir, record), 'atr')

    return Record(signals, annotations)


def trimmed_moving_average(signal, fs, type):
    """Computes trimmed moving average of the signal with sample frequency fs.
    Window size depends on the type of the signal ('ecg' or 'bp').

    Parameters:
        - signal: 1D np.array
        - fs: sample frequency
        - type: 'ecg' or 'bp' for config selection
    """
    window = np.round((W_ECG if type == 'ecg' else W_BP) * fs / 2).astype(int)
    tma = np.zeros_like(signal)
    signal = np.pad(signal, (window + 1, window + 1), 'edge')

    for i in range(window + 2, len(signal) - window):
        tma[i - window - 2] = trim_mean(signal[i - window : i + window], ALPHA / 2)

    return tma


def local_min_max(signal, fs, length, func, centered=False):
    """Computes local minimum or maximum of the signal with sample frequency fs.
    Window size length is in seconds. Non-centered window is used by default.
    Centered is only used for computing enveloped of range signal.
    """
    window = np.round(length * fs).astype(int)
    if centered:
        window = np.round(length * fs / 2).astype(int)
    local_stat = np.zeros_like(signal)

    if centered:
        signal = np.pad(signal, (window + 1, window + 1), 'edge')
        for i in range(window + 2, len(signal) - window):
            local_stat[i - window - 2] = func(signal[i - window : i + window])
    else:
        signal = np.pad(signal, (window + 1, 0), 'edge')
        for i in range(len(local_stat)):
            local_stat[i] = func(signal[i : i + window])

    return local_stat


def preprocessing(signal, fs, type):
    """It resamples the signal to F_SAMPLE, uses high-pass TMA filter to
    remove low frequencies, normalizes the signal and returns the range signal.
    """
    signal = resample_sig(signal, fs, F_SAMPLE)[0]
    tma = trimmed_moving_average(signal, F_SAMPLE, type)

    filtered = signal - tma
    filtered = filtered - np.mean(filtered)
    std = np.std(filtered)
    if std > 0:
        filtered = filtered / std

    length = L_ECG if type == 'ecg' else L_BP
    local_min = local_min_max(filtered, F_SAMPLE, length, np.min)
    local_max = local_min_max(filtered, F_SAMPLE, length, np.max)
    range_signal = local_max - local_min

    return range_signal


def beat_extraction(range_signal):
    """Function extracts beats from the range signal and returns their indices.
    together with indices of noisy parts."""
    local_min = local_min_max(range_signal, F_SAMPLE, 1.0, np.min, centered=True)
    local_max = local_min_max(range_signal, F_SAMPLE, 1.0, np.max, centered=True)
    sl_min = np.convolve(local_min, np.ones(80) / 80, mode='same')
    sl_max = np.convolve(local_max, np.ones(80) / 80, mode='same')

    noisy_parts = np.where(sl_max - sl_min <= 0.4)

    threshold = 0.5 * (sl_min + sl_max)
    # Every [c] samples have the same boolean value
    c = np.round(F_SAMPLE / 25).astype(int)

    # Beat is detected if range is above threshold at index i and below at i-1
    beats = np.repeat(range_signal[::c] > threshold[::c], c)[: len(range_signal)]
    beats = np.where(beats[1:] & ~beats[:-1])[0] + 1

    return beats, noisy_parts


def check_channel(record, channel):
    """Check if record channels are helthy and which settings should be used.
    If none of the settings are good, return empty string and discard channel.

    Parameters:
        - record: record name as string
        - channel: channel index

    Returns:
        - 'ecg', 'bp' or ''
    """
    if not globals()['dataset_dir']:
        dataset_dir = environ.get('DATASET_DIR', '')

    header = wfdb.rdheader(path.join(dataset_dir, record))
    fs = header.fs
    sampto = np.minimum(header.sig_len, fs * 30)
    record = wfdb.rdrecord(path.join(dataset_dir, record), sampto=sampto)

    p_ecg = p_bp = 0

    # Check health of channel with ECG settings
    range_signal = preprocessing(record.p_signal[:, channel], fs, 'ecg')
    beats, _ = beat_extraction(range_signal)
    if len(beats) > 1:
        rr = np.diff(beats) / F_SAMPLE
        relative_rr = rr[1:] / rr[:-1]
        p_ecg = np.mean((relative_rr >= 0.8) & (relative_rr <= 1.2))

    # Check health of channel with BP settings
    range_signal = preprocessing(record.p_signal[:, channel], fs, 'bp')
    beats, _ = beat_extraction(range_signal)
    if len(beats) > 1:
        rr = np.diff(beats) / F_SAMPLE
        relative_rr = rr[1:] / rr[:-1]
        p_bp = np.mean((relative_rr >= 0.8) & (relative_rr <= 1.2))

    if p_ecg < 0.8 and p_bp < 0.8:
        return ''
    elif p_ecg >= 0.8 and p_bp < 0.8:
        return 'ecg'
    elif p_ecg < 0.8 and p_bp >= 0.8:
        return 'bp'

    # Both settings are good, check channel name
    if any(s in header.sig_name[channel].lower() for s in BP_STRINGS):
        return 'bp'
    return 'ecg'


def detect_qrs(record):
    """Function detects QRS complexes in the signal and returns their indices.
    It uses the first channel as reference ECG signal and all others if they
    are healthy.

    Parameters:
        - record: wfdb.Record object

    Returns:
        - list of Beats objects with detected QRS complexes in healty channels
    """
    ecg = beat_extraction(preprocessing(record.p_signal[:, 0], record.fs, 'ecg'))
    beats = [Detections(0, 'ecg', *ecg)]

    for i in range(1, record.n_sig):
        sig_type = check_channel(record.record_name, i)
        if sig_type == '':
            continue

        b_n = beat_extraction(preprocessing(record.p_signal[:, i], record.fs, type))
        beats.append(Detections(i, sig_type, *b_n))

    # Because of preprocessing, beats need to be resampled to the original
    # sample frequency
    k = record.fs / F_SAMPLE
    beats = [Detections(c, y, np.round(b * k).astype(int), n) for c, y, b, n in beats]

    return beats


def correct_beats(ref_beats, beats):
    """Dynamically corrects beats in the signal using reference ECG electrode.
    Distance to the closest reference beat is computed and median filter is
    applied to smooth the signal. Corrected beats are returned.
    """
    # Since beats are sorted, we can can find the closest reference beat
    inserts = np.searchsorted(ref_beats, beats)
    inserts = np.clip(inserts, 1, len(ref_beats) - 1)

    # Check which reference beat is closer (left or right)
    diffs = np.minimum(
        np.abs(ref_beats[inserts - 1] - beats),
        np.abs(ref_beats[inserts] - beats),
    )

    delay = medfilt(diffs, 19)  # Approximately 20 seconds
    beats = np.round(beats - delay).astype(int)

    # Remove negative beats
    beats = beats[beats >= 0]

    return beats


def merge_beats(detections, fs):
    """Merge beats from different channels. If only one channel was healthy, use
    its beats. If more channels were healthy, beats first need to be checked if
    they are noisy, then they are delayed. After this, beat is safe if it is
    detected in all not noisy channels. Result is average of these beats.

    Parameters:
        - detections: list of Detections objects
        - fs: sample frequency (used to determine safe window)

    Returns:
        - np.array of merged beats
    """
    if len(detections) == 1:
        merged = detections[0].beats
        merged = merged[np.in1d(merged, detections[0].noisy_parts, invert=True)]
        return merged

    # Create list of all beats and their information
    merged = []
    for channel, _, beats, noisy_parts in detections:
        noisy_beats = np.in1d(beats, noisy_parts)
        if channel != 0:
            beats = correct_beats(detections[0].beats, beats)

        merged += [(b, channel, noisy_beats[i]) for i, b in enumerate(beats)]
    merged.sort()

    # Each beat has to be present in every not noisy channel
    i, beats = 0, []
    while i < len(merged):
        beat, channel, noisy = merged[i]
        i += 1
        if noisy:
            continue
        close_beats = [beat]
        used_channels = {channel}

        # Add all similar beats until they are too far away or channel reappears
        while i < len(merged):
            next_beat, next_channel, next_noisy = merged[i]
            if next_channel in used_channels or next_beat - beat > 0.15 * fs:
                break
            i += 1
            if not next_noisy:
                close_beats.append(next_beat)

        if len(close_beats) > 1:
            beats.append(np.mean(close_beats))

    beats = np.round(beats).astype(int)
    return beats


def write_ann_file(record, beats, ann='asc'):
    """Write annotation file with detected beats."""
    with open(record + f'.{ann}', 'w', encoding='utf-8') as f:
        for beat in beats:
            f.write(f'00:00:00.00 {beat} N 0 0 0\n')


def eval_record(record, verbose=False):
    record = read_record(record)
    detections = detect_qrs(record.record)
    beats = merge_beats(detections, record.record.fs)

    # WFDB evaluation
    fs = record.record.fs
    comparitor = Comparitor(
        record.annotations.sample, beats, np.round(0.05 * fs).astype(int)
    )
    comparitor.compare()
    if verbose:
        comparitor.print_summary()
        write_ann_file(record.record.record_name, beats)
    return comparitor


if __name__ == '__main__':
    main()

"""Scan overlapping video segments and stream their owned candidates to OCR."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
from threading import Lock

import pysrt

from backend.config import config
from backend.tools.process_manager import ProcessManager


def timestamp(milliseconds):
    return str(pysrt.SubRipTime.from_ordinal(milliseconds)).replace(',', ':')


def stop_process(process):
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def extract_segments(extractor, command, max_workers, overlap_seconds):
    duration = math.ceil(extractor.frame_count / extractor.fps * 1000)
    overlap = round(overlap_seconds * 1000)
    workers = min(max_workers, max(1, duration // (2 * overlap)))
    if workers == 1:
        return 1
    command = shlex.split(command)
    extractor.append_output(f'VSF: {workers} parallel segments, {overlap_seconds:g}s overlap')
    boundaries = [round(duration * i / workers) for i in range(workers + 1)]
    frame_margin = math.ceil(2000 / extractor.fps)
    pattern = re.compile(r'^Frame: (\d+)_(\d+)_(\d+)_(\d+)__(\d+)_(\d+)_(\d+)_(\d+)')
    manager = ProcessManager.instance()
    stopped, lock = False, Lock()
    processes = set()
    progress = [0.0] * workers

    def scan(index):
        first, last = boundaries[index:index + 2]
        start, end = max(0, first - overlap), min(duration, last + overlap)
        origin = start
        prefix = []
        retry_tail = True
        published = set()

        def enqueue(milliseconds):
            if milliseconds not in published:
                extractor.subtitle_ocr_task_queue.put((
                    extractor.frame_count, extractor._timestamp_to_frameno(milliseconds),
                    None, None, milliseconds, config.subtitleArea.value))
                published.add(milliseconds)

        while True:
            out = Path(extractor.temp_output_dir) / 'vsf_segments' / f'{index}-{start}-{end}'
            out.mkdir(parents=True, exist_ok=True)
            subtitle, log_path = out / 'raw.srt', out / 'vsf.log'
            cmd = command.copy()
            cmd[cmd.index('-o') + 1] = str(out)
            cmd[cmd.index('-ces') + 1] = str(subtitle)
            cmd += ['-s', timestamp(start)]
            if end < duration:
                cmd += ['-e', timestamp(end)]

            with log_path.open('w', encoding='utf-8') as log:
                with lock:
                    if stopped:
                        raise RuntimeError('VSF scan cancelled')
                    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.PIPE,
                                               text=True, encoding='utf-8', errors='replace',
                                               start_new_session=True)
                    processes.add(process)
                    process_id = manager.add_process(process)
                try:
                    for line in process.stderr:
                        log.write(line)
                        match = pattern.match(line)
                        if match:
                            h, m, s, ms, eh, em, es, ems = map(int, match.groups())
                            begin = ((h * 60 + m) * 60 + s) * 1000 + ms
                            finish = ((eh * 60 + em) * 60 + es) * 1000 + ems
                            if not prefix and first <= begin < last and (end == duration or finish < end - frame_margin):
                                enqueue(begin)
                            with lock:
                                progress[index] = max(progress[index], min(1, (finish - first) / (last - first)))
                                total = sum(progress) / workers * 100
                            extractor.update_progress(frame_extract=total)
                    if process.wait() != 0:
                        raise RuntimeError(f'VSF segment {index + 1} failed; see {log_path}')
                finally:
                    stop_process(process)
                    process.stderr.close()
                    with lock:
                        processes.discard(process)
                        manager.remove_process(process_id)

            found = pysrt.open(str(subtitle), encoding='utf-8')
            if prefix:
                # A shifted origin may change detection. Verify the last two
                # completed candidates before accepting any tail results.
                anchor_start, anchor_end = prefix[-2].start.ordinal, prefix[-1].end.ordinal
                expected = [(sub.start.ordinal, sub.end.ordinal) for sub in prefix[-2:]]
                actual = [(sub.start.ordinal, sub.end.ordinal) for sub in found
                          if sub.end.ordinal >= anchor_start and sub.start.ordinal <= anchor_end]
                if actual != expected:
                    start, prefix, retry_tail = origin, [], False
                    continue
                found = prefix + [sub for sub in found if sub.start.ordinal > anchor_end]
            owned = [sub for sub in found if first <= sub.start.ordinal < last]
            if end < duration and (not any(sub.end.ordinal >= last for sub in found)
                                   or any(sub.end.ordinal >= end - frame_margin for sub in owned)):
                # VSF can omit unfinished subtitles at the cutoff. Extend until
                # a completed candidate covers/passes the boundary, or reach EOF.
                complete = [sub for sub in found if sub.end.ordinal < end - frame_margin]
                if retry_tail and len(complete) >= 2:
                    start = max(origin, complete[-2].start.ordinal - overlap)
                    prefix = complete if start > origin else []
                end = min(duration, last + 2 * (end - last))
                continue
            if not published.issubset({sub.start.ordinal for sub in owned}):
                raise RuntimeError(f'VSF segment {index + 1} changed completed candidates; see {log_path}')
            for sub in owned:
                enqueue(sub.start.ordinal)
            with lock:
                progress[index] = 1.0
            return owned

    subtitles = pysrt.SubRipFile()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for future in as_completed(pool.submit(scan, index) for index in range(workers)):
                subtitles.extend(future.result())
        except BaseException:
            with lock:
                stopped = True
                active = list(processes)
            for process in active:
                stop_process(process)
            raise

    subtitles.clean_indexes()
    subtitles.save(extractor.vsf_subtitle, encoding='utf-8')
    extractor.update_progress(frame_extract=100)
    return workers

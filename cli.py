"""Extract burned-in subtitles with VSE's existing CPU/VideoSubFinder pipeline."""
import argparse
import configparser
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time


# VSE's existing configuration expresses minimum text height in 1080p units.
HEIGHT_REFERENCE = 1080


def parse_args(argv=None):
    languages = configparser.ConfigParser()
    languages.read(Path(__file__).parent / 'backend/interface/en.ini', encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('videos', type=Path, nargs='+', help='Input videos; each entire video is processed in order')
    area = parser.add_mutually_exclusive_group()
    area.add_argument('--crop', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                      help='Top-left and bottom-right corners as 0..1 ratios; origin is video top-left, x right, y down')
    area.add_argument('--crop-bottom', type=float, default=0.3, metavar='RATIO',
                      help='Fraction of full video height to keep, measured upward from the bottom; 1 keeps full frame')
    parser.add_argument('--min-text-height-ratio', type=float, default=40 / HEIGHT_REFERENCE,
                        help='Minimum box height / full video height; default: 40/1080; 0 disables filtering')
    parser.add_argument('--language', choices=list(languages['Language']), default='ch')
    parser.add_argument('--mode', choices=['fast', 'auto', 'accurate'], default='fast')
    parser.add_argument('--backend', choices=['paddle', 'openvino', 'onnxruntime'], default='paddle',
                        help='CPU OCR backend; OpenVINO uses FP32; ONNX backends require exported models')
    parser.add_argument('--threads', type=int, default=8,
                        help='Thread count for OCR and each VideoSubFinder process')
    parser.add_argument('--vsf-workers', type=int, default=6,
                        help='Parallel VSF segments on Linux/macOS; short videos use fewer; 1 disables splitting')
    parser.add_argument('--vsf-overlap', type=float, default=5.0, metavar='SECONDS',
                        help='Initial overlap around VSF segment boundaries; extended for clipped subtitles')
    parser.add_argument('--confidence', type=float, default=75.0, help='OCR confidence percent')
    parser.add_argument('--similarity', type=float, default=85.0, help='Subtitle merge similarity percent')
    parser.add_argument('--area-tolerance', type=float, default=0.0, help='VSE area overflow tolerance percent')
    parser.add_argument('--keep-work', action='store_true', help='Keep intermediate files after success; failures always retain them')
    parser.add_argument('--report', type=Path, help='Also save JSON metadata for a single input')
    args = parser.parse_args(argv)
    for name, values, high in [
        ('--crop', args.crop or [], 1),
        ('--crop-bottom', [args.crop_bottom], 1),
        ('--min-text-height-ratio', [args.min_text_height_ratio], 1),
        ('--confidence', [args.confidence], 100),
        ('--similarity', [args.similarity], 100),
        ('--area-tolerance', [args.area_tolerance], 100),
    ]:
        if any(not math.isfinite(v) or not 0 <= v <= high for v in values):
            parser.error(f'{name} must be finite and between 0 and {high}')
    if args.crop_bottom == 0:
        parser.error('--crop-bottom must be greater than 0')
    if args.crop and not (args.crop[0] < args.crop[2] and args.crop[1] < args.crop[3]):
        parser.error('--crop requires X0 < X1 and Y0 < Y1')
    if args.threads < 1:
        parser.error('--threads must be a positive integer')
    if args.vsf_workers < 1:
        parser.error('--vsf-workers must be a positive integer')
    if not math.isfinite(args.vsf_overlap) or args.vsf_overlap < 0.001:
        parser.error('--vsf-overlap must be finite and at least 0.001 seconds')
    if len(args.videos) > 1 and args.report:
        parser.error('--report requires a single input')
    return args


def resolve_area(args, width, height):
    relative = args.crop if args.crop is not None else [0, 1 - args.crop_bottom, 1, 1]
    rectangle = [round(v * extent) for v, extent in zip(relative, [width, height, width, height])]
    x0, y0, x1, y1 = rectangle
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'Subtitle area must be nonempty and inside {width}x{height}')
    return rectangle


def extract(args, video):
    video = video.expanduser().absolute()
    if not video.is_file():
        raise ValueError(f'Input video does not exist: {video}')
    output = video.with_suffix('.srt')
    work = video.with_suffix('')
    destinations = [output]
    report_path = args.report.expanduser().absolute() if args.report else None
    if args.report:
        destinations.append(report_path)
    if len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError('Output and report paths must be distinct')
    for path in destinations:
        if path.resolve().is_relative_to(work.resolve()):
            raise ValueError(f'Output must be outside the working directory: {work}')
        if path.exists() and path.samefile(video):
            raise ValueError('An output path refers to the input video')
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    previous_dir = Path.cwd()
    success = False
    alias = None
    print(f'Working directory: {work}', flush=True)
    try:
        runtime_work = work
        if platform.system() == 'Linux' and not str(work).isascii():
            # VSF under LC_ALL=C needs ASCII arguments; all data remains beside the video.
            alias = tempfile.TemporaryDirectory(prefix='vse-')
            runtime_work = Path(alias.name) / 'work'
            if not str(runtime_work).isascii():
                raise ValueError('The bundled Linux VSF needs an ASCII TMPDIR for its directory alias')
            runtime_work.symlink_to(work, target_is_directory=True)
        suffix = video.suffix if video.suffix.isascii() else '.video'
        staged = runtime_work / ('input' + suffix)
        staged.symlink_to(video)
        settings = {'Window': {'Interface': 'ch'}, 'Main': {
            'Language': args.language, 'Mode': args.mode, 'HardwareAcceleration': False,
            'OcrBackend': args.backend,
            'GenerateTxt': False, 'WordSegmentation': False, 'DebugNoDeleteCache': True,
            'VideoSubFinderCpuCores': args.threads,
            'VideoSubFinderDecoder': 'OpenCV', 'DropScore': args.confidence,
            'ThresholdTextSimilarity': args.similarity, 'SubtitleAreaDeviationRate': args.area_tolerance,
            'MinSubtitleHeight': args.min_text_height_ratio * HEIGHT_REFERENCE}}
        (work / 'config').mkdir()
        (work / 'config/config.json').write_text(json.dumps(settings), encoding='utf-8')
        os.chdir(work)
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ.setdefault('OPENCV_FOR_THREADS_NUM', '1')
        from backend.main import SubtitleExtractor
        from backend.bean.subtitle_area import SubtitleArea
        from backend.config import config
        import pysrt
        started = time.monotonic()
        extractor = SubtitleExtractor(str(staged))
        info = dict(width=extractor.frame_width, height=extractor.frame_height,
                    fps=extractor.fps, frames=extractor.frame_count)
        if not extractor.video_cap.isOpened() or not all(math.isfinite(v) and v > 0 for v in info.values()):
            raise ValueError(f'Cannot open video or read its dimensions/frame rate/count: {video}')
        info['duration_seconds'] = info['frames'] / info['fps']
        rectangle = resolve_area(args, info['width'], info['height'])
        x0, y0, x1, y1 = rectangle
        extractor.sub_area = SubtitleArea(y0, y1, x0, x1)
        extractor.temp_output_dir = str(runtime_work / 'intermediate')
        extractor.frame_output_dir = str(runtime_work / 'intermediate/frames')
        extractor.subtitle_output_dir = str(runtime_work / 'intermediate/subtitle')
        extractor.vsf_subtitle = str(runtime_work / 'intermediate/subtitle/raw_vsf.srt')
        extractor.raw_subtitle_path = str(runtime_work / 'intermediate/subtitle/raw.txt')
        extractor.subtitle_output_path = str(runtime_work / 'subtitles.srt')
        extractor.run(vsf_workers=args.vsf_workers, vsf_overlap=args.vsf_overlap)
        subs = pysrt.open(extractor.subtitle_output_path, encoding='utf-8')
        if not subs:
            raise RuntimeError('VSE produced no subtitles; inspect the retained working directory')
        report = dict(video=str(video), output=str(output), **info, area_pixels=rectangle,
                      min_text_height_pixels=config.minSubtitleHeight.value * info['height'] / HEIGHT_REFERENCE,
                      ocr_threads=config.videoSubFinderCpuCores.value or 8, vsf_threads=config.videoSubFinderCpuCores.value,
                      vsf_workers=extractor.vsf_workers, vsf_overlap_seconds=args.vsf_overlap,
                      mode=args.mode, backend=args.backend, language=args.language, cues=len(subs),
                      elapsed_seconds=time.monotonic() - started)
        (work / 'subtitles.srt').replace(output)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('w', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        success = True
        return report
    finally:
        os.chdir(previous_dir)
        if alias:
            alias.cleanup()
        if success and not args.keep_work:
            shutil.rmtree(work)
        else:
            print(f'Working directory retained: {work}', file=sys.stderr)


def main(argv=None):
    args = parse_args(argv)
    multiprocessing.set_start_method('spawn', force=True)
    reports = []
    failed = False
    for video in args.videos:
        try:
            reports.append(extract(args, video))
        except Exception as error:
            failed = True
            print(f'{video}: error: {error}', file=sys.stderr)
            reports.append({'video': str(video), 'error': str(error)})
    print(json.dumps(reports[0] if len(reports) == 1 else reports, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

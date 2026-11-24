# 추론시간, FPS, CPU·RAM(psutil) 기록과 평균·95백분위 계산.
"""프레임 CSV와 요약 JSON. 정확도/실제 모터 도달시간은 자동 측정하지 않음."""
import csv
import json
import math
import time
from pathlib import Path
from datetime import datetime
import psutil


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    pos = (len(a) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


class Metrics:
    FIELDS = ['frame', 'source_s', 'action', 'state', 'target_available',
              'capture_ms', 'detect_pipeline_ms', 'tracker_ms',
              'processing_ms', 'frame_to_command_ms', 'loop_ms',
              'x1', 'y1', 'x2', 'y2', 'command_deg', 'paused',
              'rss_mib', 'cpu_sample_percent']

    def __init__(self, output_dir, metadata):
        self.directory = Path(output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / 'frames.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.metadata = metadata
        self.process = psutil.Process()
        self.start = time.perf_counter()
        self.cpu_start = sum(self.process.cpu_times()[:2])
        self.process.cpu_percent(None)
        self.last_sample = self.start
        self.count = self.available = self.detect_count = 0
        self.times = {k: [] for k in ('detect_pipeline_ms', 'tracker_ms',
                                      'processing_ms', 'frame_to_command_ms', 'loop_ms')}
        self.rss = []
        self.closed = False

    def record(self, **row):
        now = time.perf_counter()
        row['rss_mib'] = self.process.memory_info().rss / 1024**2
        row['cpu_sample_percent'] = ''
        if now - self.last_sample >= 1:
            row['cpu_sample_percent'] = self.process.cpu_percent(None)
            self.last_sample = now
        self.writer.writerow(row)
        self.count += 1
        self.available += int(row['target_available'])
        self.detect_count += int(row['detect_pipeline_ms'] is not None)
        self.rss.append(row['rss_mib'])
        for key, values in self.times.items():
            if row.get(key) is not None:
                values.append(row[key])
        if self.count % 30 == 0:
            self.file.flush()

    def close(self, reason='finished'):
        if self.closed:
            return
        self.closed = True
        elapsed = time.perf_counter() - self.start
        self.file.close()
        stats = {}
        for key, values in self.times.items():
            stats[key] = {'mean': sum(values)/len(values) if values else None,
                          'p95': percentile(values, .95), 'samples': len(values)}
        result = {'metadata': self.metadata, 'end_reason': reason,
                  'frames': self.count, 'wall_seconds': elapsed,
                  'throughput_fps': self.count/elapsed if elapsed else None,
                  'detector_calls': self.detect_count,
                  'detector_calls_per_second': self.detect_count/elapsed if elapsed else None,
                  'target_available_fraction': self.available/self.count if self.count else None,
                  'process_cpu_percent': 100*(sum(self.process.cpu_times()[:2])-self.cpu_start)/elapsed if elapsed else None,
                  'rss_mean_mib': sum(self.rss)/len(self.rss) if self.rss else None,
                  'rss_max_sampled_mib': max(self.rss) if self.rss else None,
                  'timings': stats,
                  'notes': ['detect_pipeline_ms includes detector preprocessing and postprocessing.',
                            'CPU: 100 percent means one logical core; may exceed 100.',
                            'target_available_fraction is NOT accuracy or identity tracking success.',
                            'frame_to_command_ms starts after frame delivery, excludes sensor/buffer age and mechanical movement.',
                            'loop_ms excludes metrics.record; throughput includes logging/display overhead.',
                            'RSS is sampled once per processed frame; not an absolute peak.',
                            'Video processing is offline, all frames, no real-time pacing.']}
        (self.directory / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Results: {self.directory}')

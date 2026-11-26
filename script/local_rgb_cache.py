"""Optional local RGB cache. Keeps original dataset and training equations unchanged."""
import hashlib
import json
import os
from pathlib import Path
import av
import numpy as np
from dataset.taf_dataset import TaFDataset

class CachedTaFDataset(TaFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rgb_maps = {}
        root = Path(os.environ['TAF_RGB_CACHE'])
        root.mkdir(parents=True, exist_ok=True)
        for ep in self.episodes:
            key = str(ep.video.resolve())
            if key in self._rgb_maps:
                continue
            stat = ep.video.stat()
            signature = json.dumps([key, stat.st_size, stat.st_mtime_ns, self.image_size])
            stem = hashlib.sha256(signature.encode()).hexdigest()
            image_path, pts_path = root/f'{stem}.npy', root/f'{stem}_pts.npy'
            # Commit marker is the PTS file; an interrupted image array is rebuilt.
            if not (image_path.exists() and pts_path.exists()):
                print(f'Building exact RGB cache: {ep.video}', flush=True)
                with av.open(str(ep.video)) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = 'NONE'
                    stream.codec_context.thread_count = 1
                    pts = []
                    # Frame count is probed independently; container metadata may omit it.
                    from dataset.taf_dataset import _probe_video_timestamps
                    expected_times = _probe_video_timestamps(ep.video)
                    size = self.image_size
                    if size is None:
                        size = (stream.height, stream.width)
                    elif isinstance(size, int):
                        size = (size, size)
                    array = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8,
                        shape=(len(expected_times), int(size[0]), int(size[1]), 3))
                    for i, frame in enumerate(container.decode(stream)):
                        if i >= len(expected_times): raise ValueError('Extra video frame')
                        rgb = frame.reformat(width=int(size[1]),height=int(size[0]),format='rgb24',interpolation='AREA').to_ndarray()
                        array[i] = rgb
                        pts.append(frame.pts)
                        if (i+1) % 10000 == 0: print(f'Cached {i+1}/{len(expected_times)} frames',flush=True)
                    pts = np.asarray(pts,dtype=np.int64)
                    assert np.array_equal(pts,self._timestamps_to_pts(expected_times,stream.time_base))
                    array.flush(); del array
                    np.save(pts_path,pts)
            self._rgb_maps[key] = (np.load(image_path,mmap_mode='r'),np.load(pts_path,mmap_mode='r'))
        # Compare to the original random-access decoder, independently of the cache.
        rng = np.random.default_rng(42)
        for ep in self.episodes:
            start = int(rng.integers(0,ep.length-self.window_size+1))
            offsets = np.arange(start,start+self.window_size,dtype=np.int64)
            expected, expected_pts = super()._images(ep,offsets)
            actual, actual_pts = self._images(ep,offsets)
            assert np.array_equal(expected,actual), 'RGB cache mismatch'
            assert np.array_equal(expected_pts,actual_pts), 'PTS cache mismatch'
        print(f'RGB cache equivalence PASS: {len(self.episodes)} random episode windows, pixel-exact and PTS-exact',flush=True)

    def _images(self,ep,offsets):
        arrays = getattr(self,'_rgb_maps',{})
        key = str(ep.video.resolve())
        if key not in arrays:
            return super()._images(ep,offsets)
        frames, pts = arrays[key]
        indices = ep.video_frame_indices[offsets]
        return np.asarray(frames[indices]).copy(), np.asarray(pts[indices]).copy()

    def _sensor_rows(self, ep, offsets):
        # The upstream one-file cache thrashes when a batch mixes sequences.
        if not hasattr(self, '_sensor_maps'):
            self._sensor_maps = {}
        self._sensor_cache = self._sensor_maps.get(str(ep.parquet))
        result = super()._sensor_rows(ep, offsets)
        self._sensor_maps[str(ep.parquet)] = self._sensor_cache
        return result

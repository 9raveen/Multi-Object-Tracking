from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=60, n_init=3):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.7,
            max_cosine_distance=0.2,
            nn_budget=100
        )

    def update(self, detections, frame):
        """
        Args:
            detections: np.ndarray (N, 5) -> [x1, y1, x2, y2, conf]
            frame: current frame

        Returns:
            list of [x1, y1, x2, y2, track_id]
        """

        ds_detections = []

        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf = map(float, det)

                w = x2 - x1
                h = y2 - y1

                ds_detections.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []
        h, w = frame.shape[:2]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Clamp to frame
            l = max(0, min(l, w))
            r = max(0, min(r, w))
            t = max(0, min(t, h))
            b = max(0, min(b, h))

            results.append([l, t, r, b, track_id])

        return results
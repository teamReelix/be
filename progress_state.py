from threading import Lock

progress_data = {
    "total": 0,
    "done": 0,
    "current_start": 0
}

progress_lock = Lock()
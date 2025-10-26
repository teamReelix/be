import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def set_progress(job_id: str, total: int, done: int):
    r.hset(f"progress:{job_id}", mapping={"total": total, "done": done})

def get_progress(job_id: str):
    data = r.hgetall(f"progress:{job_id}")
    if not data:
        return {"total": 0, "done": 0}
    return {"total": int(data["total"]), "done": int(data["done"])}

def increment_done(job_id: str, count: int = 1):
    r.hincrby(f"progress:{job_id}", "done", count)

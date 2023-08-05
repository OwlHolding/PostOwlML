from multiprocessing import cpu_count

bind = "127.0.0.1:8000"


workers = cpu_count() + 1
worker_class = 'uvicorn.workers.UvicornWorker'
loglevel = 'debug'
accesslog = '/home/data/PostOwlML/access_log'
errorlog = '/home/data/PostOwlML/error_log'

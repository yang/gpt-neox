import torch, torch.cuda


class Metric:
    """
    Dumb utility to collect and report average wall-time metrics.
    """

    def __init__(self, label):
        self.label = label
        self.measurements = []

    def collect(self, measurement):
        self.measurements.append(measurement)

    def get_measurements(self):
        return self.measurements[:]

    def report(self):
        print(
            self.label,
            torch.quantile(torch.tensor(self.measurements), torch.arange(10) / 10.0),
        )


def monitor_method_cuda_wall_times(metric, obj, methodname):
    """
    Measure timings for a method on an object or class.

    For instance:

    >>> metric = Metric('!LNORM')
    >>> monitor_method_wall_times(metric, LayerNorm, 'forward')
    """
    oldmeth = getattr(obj, methodname)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def newmeth(*args, **kw):
        start_event.record()
        try:
            return oldmeth(*args, **kw)
        finally:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            metric.collect(elapsed)
            metric.report()

    setattr(obj, methodname, newmeth)


import pprint


def safeprint(*args):
    import fcntl

    with open("/dev/null", "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        rank = torch.distributed.get_rank()
        print(
            f"!! {rank=}",
            *[a if type(a) is str else pprint.pformat(a, width=200) for a in args],
        )


def safeprint0(*args):
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            f"!!",
            *[a if type(a) is str else pprint.pformat(a, width=200) for a in args],
        )

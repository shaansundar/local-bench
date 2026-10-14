from bench.tau import TauBenchmark


class Telecom(TauBenchmark):
    domain = "telecom"
    split = "base"


BENCHMARK = Telecom

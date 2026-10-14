from bench.tau import TauBenchmark


class Banking(TauBenchmark):
    domain = "banking_knowledge"

    @property
    def extra_args(self):
        return ["--retrieval-config", self.spec.options.get("retrieval_config", "bm25")]


BENCHMARK = Banking

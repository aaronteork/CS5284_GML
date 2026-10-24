"""Parallel dispatch of karateclub's unchanged BiasedRandomWalker.do_walk."""
import importlib
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial

import numpy as np
from karateclub.utils.walker import BiasedRandomWalker, _get_edge_fn, _get_weight_fn
from tqdm.auto import tqdm

CHUNK_NODES = 128
_worker_walker = None
_worker_seed = None


def _init_worker(graph, walk_length, walk_number, p, q, seed):
    global _worker_walker, _worker_seed
    _worker_walker = BiasedRandomWalker(walk_length, walk_number, p, q)
    _worker_walker.graph = graph
    _worker_walker.edge_fn = _get_edge_fn(graph)
    _worker_walker.weight_fn = _get_weight_fn(graph)
    _worker_seed = seed


def _walk_chunk(job):
    chunk_id, nodes = job
    # Per-chunk seeds and ordered results keep walks independent of scheduling.
    seed = int(np.random.SeedSequence([_worker_seed, chunk_id]).generate_state(1)[0])
    np.random.seed(seed)
    random.seed(seed)
    return [
        _worker_walker.do_walk(node)
        for node in nodes
        for _ in range(_worker_walker.walk_number)
    ]


class _ParallelWalker(BiasedRandomWalker):
    def __init__(self, walk_length, walk_number, p, q, *, workers, seed):
        super().__init__(walk_length, walk_number, p, q)
        self.workers = workers
        self.seed = seed

    def do_walks(self, graph):
        # Node2Vec.fit has already applied karateclub's graph checks/self-loops.
        nodes = list(graph.nodes())
        jobs = ((i, nodes[start:start + CHUNK_NODES])
                for i, start in enumerate(range(0, len(nodes), CHUNK_NODES)))
        self.walks = []
        print(f"Generating {len(nodes) * self.walk_number:,} original karateclub walks "
              f"using {self.workers} processes", flush=True)
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(graph, self.walk_length, self.walk_number, self.p, self.q, self.seed),
        ) as pool, tqdm(total=len(nodes), desc="Node2Vec walks", unit="node") as progress:
            for walks in pool.map(_walk_chunk, jobs, chunksize=1):
                self.walks.extend(walks)
                progress.update(len(walks) // self.walk_number)
        print("Walk generation complete; starting original gensim embedding training", flush=True)


@contextmanager
def parallel_node2vec_walks(workers, seed):
    """Change only walk dispatch inside the original Node2Vec.fit call."""
    module = importlib.import_module("karateclub.node_embedding.neighbourhood.node2vec")
    original = module.BiasedRandomWalker
    module.BiasedRandomWalker = partial(_ParallelWalker, workers=workers, seed=seed)
    try:
        yield
    finally:
        module.BiasedRandomWalker = original

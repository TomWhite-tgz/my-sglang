from __future__ import annotations

import argparse
import heapq
import itertools
import unittest
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


class TokenToKVPool:
    def __init__(self, size, num_heads, head_dim, device="cpu"):
        if min(size, num_heads, head_dim) <= 0:
            raise ValueError("容量和注意力维度必须为正数")
        self.size = size
        self.free_slots = deque(range(size))
        self.allocated = set()
        self.kv_buffer = torch.zeros((size, 2, num_heads, head_dim), device=device)

    def alloc(self, n: int) -> List[int]:
        if n > len(self.free_slots):
            raise MemoryError("KV 槽位不足")
        slots = [self.free_slots.popleft() for _ in range(n)]
        self.allocated.update(slots)
        return slots

    def free(self, slots: List[int]):
        if any(s not in self.allocated for s in slots) or len(set(slots)) != len(slots):
            raise ValueError(f"槽位 {slots} 未分配或重复释放")
        self.allocated.difference_update(slots)
        self.free_slots.extend(slots)


class RadixNode:
    _ids = itertools.count()

    def __init__(self, token_ids, slots, parent=None, last_access=0):
        self.token_ids, self.slots = list(token_ids), list(slots)
        self.parent: Optional[RadixNode] = parent
        self.children: Dict[int, RadixNode] = {}
        self.ref_count = 0
        self.last_access = last_access
        self.id = next(RadixNode._ids)

    def __lt__(self, other):
        return (self.last_access, self.id) < (other.last_access, other.id)


class RadixCache:
    def __init__(self, pool: TokenToKVPool, verbose=False):
        self.pool, self.verbose = pool, verbose
        self.root = RadixNode([], [])
        self.clock = 0

    def nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            yield node

    def _split(self, node: RadixNode, k: int) -> RadixNode:
        prefix = RadixNode(
            node.token_ids[:k], node.slots[:k], node.parent, node.last_access
        )
        prefix.ref_count = node.ref_count
        node.parent.children[prefix.token_ids[0]] = prefix
        node.token_ids, node.slots, node.parent = (
            node.token_ids[k:],
            node.slots[k:],
            prefix,
        )
        prefix.children[node.token_ids[0]] = node
        return prefix

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], RadixNode]:
        self.clock += 1
        node, slots = self.root, []
        while len(slots) < len(tokens):
            child = node.children.get(tokens[len(slots)])
            if child is None:
                break
            rest, k = tokens[len(slots) :], 0
            while (
                k < min(len(child.token_ids), len(rest))
                and child.token_ids[k] == rest[k]
            ):
                k += 1
            if k < len(child.token_ids):
                child = self._split(child, k)
            child.last_access = self.clock
            slots += child.slots
            node = child
        return slots, node

    def insert(self, tokens, slots) -> Tuple[RadixNode, List[int], List[int]]:
        if len(tokens) != len(slots):
            raise ValueError("每个 token 必须对应一个 KV 槽位")
        cached, node = self.match_prefix(tokens)
        k = len(cached)
        duplicates = [s for c, s in zip(cached, slots) if c != s]
        if k < len(tokens):
            child = RadixNode(tokens[k:], slots[k:], node, self.clock)
            node.children[tokens[k]] = child
            node = child
        return node, cached + list(slots[k:]), duplicates

    def lock_path(self, node: RadixNode):
        while node is not self.root:
            node.ref_count += 1
            node = node.parent

    def unlock_path(self, node: RadixNode):
        path = []
        while node is not self.root:
            path.append(node)
            node = node.parent
        if any(n.ref_count <= 0 for n in path):
            raise RuntimeError("解除了一条未被保护的路径")
        for n in path:
            n.ref_count -= 1

    def evictable_size(self) -> int:
        return sum(
            len(n.slots)
            for n in self.nodes()
            if n is not self.root and n.ref_count == 0
        )

    def evict(self, num_tokens: int):
        heap = [
            n
            for n in self.nodes()
            if n is not self.root and not n.children and n.ref_count == 0
        ]
        heapq.heapify(heap)
        freed = 0
        while freed < num_tokens and heap:
            node = heapq.heappop(heap)
            self.pool.free(node.slots)
            freed += len(node.slots)
            parent = node.parent
            del parent.children[node.token_ids[0]]
            if (
                parent is not self.root
                and not parent.children
                and parent.ref_count == 0
            ):
                heapq.heappush(heap, parent)
            if self.verbose:
                print(f"  [LRU] 淘汰 {node.token_ids}")
        if freed < num_tokens:
            raise MemoryError("没有足够的可淘汰槽位：其余槽位由活跃请求持有")

    def print_tree(self, node=None, indent=""):
        node = node or self.root
        for child in node.children.values():
            print(
                f"{indent}+ {child.token_ids} -> slots {child.slots} (ref={child.ref_count})"
            )
            self.print_tree(child, indent + "  ")


@dataclass
class Req:
    rid: int
    prompt: List[int]
    max_new_tokens: int
    output: List[int] = field(default_factory=list)
    kv_indices: List[int] = field(default_factory=list)
    last_node: Optional[RadixNode] = None
    prefix_len: int = 0
    logits: Optional[torch.Tensor] = None
    logits_history: List[torch.Tensor] = field(default_factory=list)

    def reserve(self) -> int:
        return len(self.prompt) + self.max_new_tokens - 1 - len(self.kv_indices)


class NanoRadixEngine:
    def __init__(
        self,
        num_slots=32,
        num_heads=2,
        head_dim=16,
        vocab_size=128,
        max_running=8,
        use_cache=True,
        device="cpu",
        verbose=True,
    ):
        if vocab_size <= 1 or max_running <= 0:
            raise ValueError("vocab_size 必须大于 1，max_running 必须为正")
        self.num_heads, self.head_dim, self.vocab_size = num_heads, head_dim, vocab_size
        self.max_running, self.use_cache, self.verbose = max_running, use_cache, verbose
        self.pool = TokenToKVPool(num_slots, num_heads, head_dim, device)
        self.cache = RadixCache(self.pool, verbose)
        self.waiting: List[Req] = []
        self.running: List[Req] = []
        self.finished: List[Req] = []
        self._rid = itertools.count()
        self.device = self.pool.kv_buffer.device
        dim = num_heads * head_dim
        rng = torch.Generator().manual_seed(42)
        self.embedding = torch.randn(vocab_size, dim, generator=rng).to(self.device)
        self.wq, self.wk, self.wv, self.lm_head = [
            (torch.randn(dim, out, generator=rng) / dim**0.5).to(self.device)
            for out in (dim, dim, dim, vocab_size)
        ]
        self.frequencies = 1 / (10000 ** (torch.arange(dim, device=self.device) / dim))

    def _forward_extend(self, tokens: List[int], start: int, kv_indices: List[int]):
        n = len(tokens)
        pos = torch.arange(start, start + n, device=self.device)
        x = self.embedding[torch.tensor(tokens, device=self.device)]
        x = x + torch.sin(pos[:, None] * self.frequencies)
        q, k, v = [
            (x @ w).view(n, self.num_heads, self.head_dim)
            for w in (self.wq, self.wk, self.wv)
        ]
        idx = torch.tensor(kv_indices, dtype=torch.long, device=self.device)
        self.pool.kv_buffer[idx[start:], 0] = k
        self.pool.kv_buffer[idx[start:], 1] = v
        keys, values = self.pool.kv_buffer[idx, 0], self.pool.kv_buffer[idx, 1]
        scores = torch.einsum("nhd,shd->hns", q, keys) / self.head_dim**0.5
        causal = torch.arange(len(idx), device=self.device)[None, :] > pos[:, None]
        scores = scores.masked_fill(causal, float("-inf"))
        out = torch.einsum("hns,shd->nhd", scores.softmax(-1), values)
        return ((out.reshape(n, -1) + x) @ self.lm_head)[-1]

    def add_request(self, prompt: List[int], max_new_tokens=3) -> Req:
        prompt = list(prompt)
        if not prompt:
            raise ValueError("prompt 不能为空")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须为非负整数")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in prompt):
            raise ValueError("prompt token 必须是词表范围内的整数")
        req = Req(next(self._rid), prompt, max_new_tokens)
        if max_new_tokens == 0:
            return req
        if req.reserve() > self.pool.size:
            raise ValueError(f"请求需要 {req.reserve()} 个槽位，容量只有 {self.pool.size}")
        self.waiting.append(req)
        return req

    def _alloc(self, n: int) -> List[int]:
        shortfall = n - len(self.pool.free_slots)
        if shortfall > 0:
            self.cache.evict(shortfall)
        return self.pool.alloc(n)

    def _schedule(self) -> List[Req]:
        for req in self.waiting:
            req.kv_indices, req.last_node = (
                self.cache.match_prefix(req.prompt[:-1])
                if self.use_cache
                else ([], self.cache.root)
            )
        self.waiting.sort(key=lambda r: -len(r.kv_indices))
        batch = []
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            self.cache.lock_path(req.last_node)

            available = len(self.pool.free_slots) + self.cache.evictable_size()
            if req.reserve() + sum(r.reserve() for r in self.running) > available:
                self.cache.unlock_path(req.last_node)
                break
            self.waiting.pop(0)
            req.prefix_len = len(req.kv_indices)
            self.running.append(req)
            batch.append(req)
        return batch

    def _cache_req(self, req: Req, finished: bool):
        if not self.use_cache:
            if finished:
                self.pool.free(req.kv_indices)
            return
        tokens = (req.prompt + req.output)[: len(req.kv_indices)]
        node, req.kv_indices, duplicates = self.cache.insert(tokens, req.kv_indices)

        self.pool.free(duplicates)
        if not finished:
            self.cache.lock_path(node)
        self.cache.unlock_path(req.last_node)
        req.last_node = node

    def _prefill(self, req: Req):
        start = len(req.kv_indices)
        req.kv_indices += self._alloc(len(req.prompt) - start)
        req.logits = self._forward_extend(req.prompt[start:], start, req.kv_indices)
        self._cache_req(req, finished=False)
        if self.verbose:
            print(
                f"[prefill] rid={req.rid} {req.prompt} 复用 {req.prefix_len}/{len(req.prompt)}"
            )

    def _decode(self):
        for req in list(self.running):
            req.logits_history.append(req.logits)
            req.output.append(int(req.logits.argmax()))
            if len(req.output) == req.max_new_tokens:
                self._cache_req(req, finished=True)
                self.running.remove(req)
                self.finished.append(req)
                if self.verbose:
                    print(f"[finish]  rid={req.rid} 输出 {req.output}")
                continue
            req.kv_indices += self._alloc(1)
            req.logits = self._forward_extend(
                [req.output[-1]], len(req.kv_indices) - 1, req.kv_indices
            )

    def _abort_all(self):
        cached = {s for n in self.cache.nodes() for s in n.slots}
        for req in self.running:
            self.cache.unlock_path(req.last_node)
            self.pool.free([s for s in req.kv_indices if s not in cached])
        self.running, self.waiting = [], []

    @torch.inference_mode()
    def run(self):
        try:
            while self.waiting or self.running:
                for req in self._schedule():
                    self._prefill(req)
                self._decode()
        except BaseException:
            self._abort_all()
            raise

    def generate(self, prompt, max_new_tokens=3) -> List[int]:
        req = self.add_request(prompt, max_new_tokens)
        self.run()
        return req.output


class CacheTests(unittest.TestCase):
    PROMPTS = ([1, 2, 3, 4], [1, 2, 3, 5], [1, 2], [8, 9, 10, 11, 12, 13])

    def make_engine(self, **kw):
        return NanoRadixEngine(num_heads=2, head_dim=4, verbose=False, **kw)

    def reference(self, prompt, n=3):
        return self.make_engine(use_cache=False).generate(prompt, n)

    def assert_clean(self, engine):
        self.assertFalse(engine.waiting or engine.running)
        cached = []
        for node in engine.cache.nodes():
            self.assertEqual(node.ref_count, 0)
            self.assertEqual(len(node.token_ids), len(node.slots))
            cached += node.slots
            for token, child in node.children.items():
                self.assertIs(child.parent, node)
                self.assertEqual(token, child.token_ids[0])
        self.assertEqual(len(cached), len(set(cached)))
        self.assertEqual(set(cached), engine.pool.allocated)
        free = list(engine.pool.free_slots)
        self.assertEqual(len(free), len(set(free)))
        self.assertEqual(
            set(free) | engine.pool.allocated, set(range(engine.pool.size))
        )
        self.assertFalse(set(free) & engine.pool.allocated)

    def test_branch_and_prefix_hit(self):
        engine = self.make_engine()
        engine.generate([1, 2, 3, 4], 2)
        req = engine.add_request([1, 2, 3, 5], 2)
        engine.run()
        self.assertEqual(req.prefix_len, 3)
        engine.generate([1, 2], 2)
        self.assert_clean(engine)

    def test_repeated_request_has_no_leak(self):
        engine = self.make_engine()
        expected = engine.generate([1, 2, 3, 4], 3)
        free = len(engine.pool.free_slots)
        for _ in range(3):
            self.assertEqual(engine.generate([1, 2, 3, 4], 3), expected)
            self.assertEqual(len(engine.pool.free_slots), free)
        self.assert_clean(engine)

    def test_cached_matches_uncached(self):
        engine = self.make_engine()
        engine.generate([1, 2, 3, 4], 3)
        for prompt in self.PROMPTS:
            ref = self.make_engine(use_cache=False)
            a, b = engine.add_request(prompt, 3), ref.add_request(prompt, 3)
            engine.run()
            ref.run()
            self.assertEqual(a.output, b.output)
            for x, y in zip(a.logits_history, b.logits_history):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
        self.assert_clean(engine)

    def test_batch_with_identical_prompts_deduplicates(self):
        engine = self.make_engine()
        reqs = [engine.add_request(p, 3) for p in ([1, 2, 3], [1, 2, 3], [1, 2, 7])]
        engine.run()
        for req in reqs:
            self.assertEqual(req.output, self.reference(req.prompt))
        self.assert_clean(engine)

    def test_longest_prefix_first(self):
        engine = self.make_engine(max_running=1)
        engine.generate([1, 2, 3, 4], 2)
        miss = engine.add_request([9, 9, 9], 2)
        hit = engine.add_request([1, 2, 3, 4, 5], 2)
        engine.run()
        self.assertEqual(engine.finished[-2:], [hit, miss])
        self.assert_clean(engine)

    def test_admission_and_eviction_under_pressure(self):
        engine = self.make_engine(num_slots=8)
        reqs = [engine.add_request(p, 3) for p in self.PROMPTS]
        engine.run()
        for req in reqs:
            self.assertEqual(req.output, self.reference(req.prompt))
        self.assert_clean(engine)

    def test_lru_refreshes_on_hit(self):
        engine = self.make_engine(num_slots=2)
        cache, pool = engine.cache, engine.pool
        cache.insert([1], pool.alloc(1))
        cache.insert([2], pool.alloc(1))
        cache.match_prefix([1])
        cache.evict(1)
        self.assertEqual(set(cache.root.children), {1})
        self.assert_clean(engine)

    def test_split_preserves_active_lock(self):
        engine = self.make_engine(num_slots=3)
        suffix = engine.cache.insert([1, 2, 3], engine.pool.alloc(3))[0]
        engine.cache.lock_path(suffix)
        prefix = engine.cache.match_prefix([1])[1]
        self.assertIs(suffix.parent, prefix)
        self.assertEqual((prefix.ref_count, suffix.ref_count), (1, 1))
        with self.assertRaises(MemoryError):
            engine.cache.evict(1)
        engine.cache.unlock_path(suffix)
        engine.cache.evict(3)
        self.assert_clean(engine)

    def test_failure_cleans_slots_and_locks(self):
        engine = self.make_engine()
        engine.generate([1, 2, 3, 4], 2)
        original, calls = engine._forward_extend, itertools.count()

        def flaky(*args):
            if next(calls) == 3:
                raise RuntimeError("模拟模型执行失败")
            return original(*args)

        engine._forward_extend = flaky
        for prompt in ([1, 2, 3, 5], [1, 2, 9], [7, 7]):
            engine.add_request(prompt, 3)
        with self.assertRaises(RuntimeError):
            engine.run()
        self.assert_clean(engine)
        engine._forward_extend = original
        engine.generate([1, 2, 3, 5], 3)
        self.assert_clean(engine)

    def test_validation_and_double_free(self):
        engine = self.make_engine(num_slots=4)
        for prompt, n in (
            ([], 1),
            ([1], -1),
            ([1], 1.5),
            ([128], 1),
            ([1, 2, 3, 4], 2),
        ):
            with self.assertRaises(ValueError):
                engine.add_request(prompt, n)
        self.assertEqual(engine.generate([1], 0), [])
        slots = engine.pool.alloc(1)
        engine.pool.free(slots)
        with self.assertRaises(ValueError):
            engine.pool.free(slots)
        self.assert_clean(engine)


def demo():
    engine = NanoRadixEngine(num_slots=16)
    engine.generate([1, 2, 3, 4], 2)

    for prompt in ([1, 2, 3, 5], [1, 2, 3, 5], [1, 2, 9], [8, 9, 10, 11, 12, 13]):
        engine.add_request(prompt, 3)
    engine.run()
    print("\n[Radix tree]")
    engine.cache.print_tree()
    print(f"空闲槽位: {len(engine.pool.free_slots)}/{engine.pool.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        unittest.main(argv=["nano_radix.py"], verbosity=2)
    else:
        demo()

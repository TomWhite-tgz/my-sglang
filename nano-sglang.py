import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


    """管理离散的显存块 (Paged KV Cache Blocks)"""
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, device="cpu"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        
        # 预先分配物理 KV 缓存池: [num_blocks, 2 (K and V), block_size, num_heads, head_dim]
        # 注：为了让初版极致简单，本实现中设置 block_size = 1（Token 级物理分页）
        self.kv_buffer = torch.zeros(
            (num_blocks, 2, block_size, num_heads, head_dim), 
            dtype=torch.float32, device=device
        )

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of Memory: 显存块已耗尽，需要触发 LRU 淘汰！")
        return self.free_blocks.pop(0)

    def free(self, block_id: int):
        self.free_blocks.append(block_id)


class RadixNode:
    def __init__(self, token_ids: List[int], block_ids: List[int], parent=None):
        self.token_ids = token_ids          # 该节点缓存的连续 Token 序列
        self.block_ids = block_ids          # 对应的物理 Block ID 列表
        self.parent: Optional[RadixNode] = parent
        self.children: Dict[int, RadixNode] = {}  # 首 Token ID -> 子节点
        self.ref_count = 0                  # 引用计数 (正在被推理使用的请求数)
        self.last_access_time = time.time()

    def is_leaf(self):
        return len(self.children) == 0

class RadixCache:
    def __init__(self, block_manager: BlockManager):
        self.bm = block_manager
        self.root = RadixNode(token_ids=[], block_ids=[])

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], List[int], RadixNode, int]:
        curr = self.root
        matched_tokens = []
        matched_blocks = []
        idx = 0

        while idx < len(tokens):
            first_token = tokens[idx]
            if first_token not in curr.children:
                break
            child = curr.children[first_token]
            
            child_tokens = child.token_ids
            match_len = 0
            while (idx + match_len < len(tokens) and 
                   match_len < len(child_tokens) and 
                   tokens[idx + match_len] == child_tokens[match_len]):
                match_len += 1

            matched_tokens.extend(child.token_ids[:match_len])
            matched_blocks.extend(child.block_ids[:match_len])
            idx += match_len
            curr = child

            if match_len < len(child_tokens):
                return matched_tokens, matched_blocks, curr, match_len

        return matched_tokens, matched_blocks, curr, len(curr.token_ids)

    def split_node(self, node: RadixNode, split_idx: int) -> RadixNode:
        child_tokens = node.token_ids[split_idx:]
        child_blocks = node.block_ids[split_idx:]
        
        new_child = RadixNode(child_tokens, child_blocks, parent=node)
        new_child.children = node.children
        for c in new_child.children.values():
            c.parent = new_child

        node.token_ids = node.token_ids[:split_idx]
        node.block_ids = node.block_ids[:split_idx]
        node.children = {child_tokens[0]: new_child}
        return node

    def insert(self, tokens: List[int], block_ids: List[int]) -> RadixNode:
        _, _, last_node, match_offset = self.match_prefix(tokens)
        
        if match_offset < len(last_node.token_ids):
            last_node = self.split_node(last_node, match_offset)

        remaining_tokens = tokens[len(last_node.token_ids):] # 简化写法，实际按累计长度切分
        consumed_len = sum(len(n.token_ids) for n in self._get_ancestors(last_node))
        remaining_tokens = tokens[consumed_len:]
        remaining_blocks = block_ids[consumed_len:]

        if remaining_tokens:
            new_node = RadixNode(remaining_tokens, remaining_blocks, parent=last_node)
            last_node.children[remaining_tokens[0]] = new_node
            return new_node
        return last_node

    def _get_ancestors(self, node: RadixNode) -> List[RadixNode]:
        res = []
        curr = node
        while curr is not None:
            res.append(curr)
            curr = curr.parent
        return res[::-1]

    def evict(self, num_blocks_needed: int):
        while len(self.bm.free_blocks) < num_blocks_needed:
            leaves = []
            self._collect_evictable_leaves(self.root, leaves)
            if not leaves:
                raise RuntimeError("显存彻底耗尽，且所有前缀均处于活跃引用中，无法淘汰！")
            
            leaves.sort(key=lambda x: x.last_access_time)
            target = leaves[0]
            
            for bid in target.block_ids:
                self.bm.free(bid)
            
            parent = target.parent
            del parent.children[target.token_ids[0]]
            print(f" [LRU 淘汰] 释放了节点 Token: {target.token_ids}, 归还 {len(target.block_ids)} 个物理 Block")

    def _collect_evictable_leaves(self, curr: RadixNode, leaves: List[RadixNode]):
        if curr != self.root and curr.is_leaf() and curr.ref_count == 0:
            leaves.append(curr)
        for child in curr.children.values():
            self._collect_evictable_leaves(child, leaves)

    def print_tree(self, node: Optional[RadixNode] = None, indent=""):
        if node is None:
            node = self.root
            print("\n====== Radix Tree 状态可视化 ======")
            print("[Root]")
        for first_token, child in node.children.items():
            ref_str = f" [Ref: {child.ref_count}]" if child.ref_count > 0 else ""
            print(f"{indent}└── Tokens: {child.token_ids} -> Blocks: {child.block_ids}{ref_str}")
            self.print_tree(child, indent + "    ")


class NanoSGLangEngine:
    def __init__(self, num_blocks=32, num_heads=2, head_dim=16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bm = BlockManager(num_blocks=num_blocks, block_size=1, num_heads=num_heads, head_dim=head_dim)
        self.cache = RadixCache(self.bm)

    def _toy_forward_mha(self, q, k, v, block_table: List[int]):

        curr_block_id = block_table[-1]
        self.bm.kv_buffer[curr_block_id, 0, 0] = k
        self.bm.kv_buffer[curr_block_id, 1, 0] = v

        all_blocks = torch.tensor(block_table, dtype=torch.long)
        keys = self.bm.kv_buffer[all_blocks, 0, 0]
        values = self.bm.kv_buffer[all_blocks, 1, 0]

        scores = torch.einsum("hd, shd -> hs", q, keys) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.einsum("hs, shd -> hd", attn_weights, values)
        return out

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 3):
        print(f"\n>>> 收到请求，Prompt: {prompt_tokens}")
        hit_tokens, hit_blocks, last_node, offset = self.cache.match_prefix(prompt_tokens)
        hit_len = len(hit_tokens)
        hit_rate = (hit_len / len(prompt_tokens)) * 100
        print(f"[*] Cache 命中: {hit_len}/{len(prompt_tokens)} tokens (命中率: {hit_rate:.1f}%) | 复用 Block: {hit_blocks}")

        curr_ancestors = self.cache._get_ancestors(last_node)
        for a in curr_ancestors:
            a.ref_count += 1

        allocated_blocks = list(hit_blocks)
        uncached_tokens = prompt_tokens[hit_len:]

        for t in uncached_tokens:
            self.cache.evict(num_blocks_needed=1)
            new_block = self.bm.allocate()
            allocated_blocks.append(new_block)
            
            q = torch.randn(self.num_heads, self.head_dim)
            k = torch.randn(self.num_heads, self.head_dim)
            v = torch.randn(self.num_heads, self.head_dim)
            self._toy_forward_mha(q, k, v, allocated_blocks[:len(allocated_blocks)])

        leaf_node = self.cache.insert(prompt_tokens, allocated_blocks)
        leaf_node.ref_count += 1

        generated = []
        curr_tokens = list(prompt_tokens)
        for step in range(max_new_tokens):
            self.cache.evict(num_blocks_needed=1)
            new_block = self.bm.allocate()
            allocated_blocks.append(new_block)

            q = torch.randn(self.num_heads, self.head_dim)
            k = torch.randn(self.num_heads, self.head_dim)
            v = torch.randn(self.num_heads, self.head_dim)
            _ = self._toy_forward_mha(q, k, v, allocated_blocks)

            next_token = 100 + step
            generated.append(next_token)
            curr_tokens.append(next_token)

        final_node = self.cache.insert(curr_tokens, allocated_blocks)

        for a in curr_ancestors:
            a.ref_count -= 1
        leaf_node.ref_count -= 1

        print(f"[✓] 生成完毕，输出 Tokens: {generated}")
        return generated


if __name__ == "__main__":

    engine = NanoSGLangEngine(num_blocks=16)


    req1 = [1, 2, 3, 4]
    engine.generate(req1, max_new_tokens=2)
    engine.cache.print_tree()


    req2 = [1, 2, 3, 5] 
    engine.generate(req2, max_new_tokens=2)
    engine.cache.print_tree()


    print("\n--- 注入大请求，观察树被 LRU 自动剪枝 ---")
    req3 = [8, 9, 10, 11, 12, 13]
    engine.generate(req3, max_new_tokens=4)
    engine.cache.print_tree()
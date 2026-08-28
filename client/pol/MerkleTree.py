"""
Merkle树实现
用于生成PoL承诺和验证证明
"""

import hashlib
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MerkleTree:
    """
    Merkle树实现

    用于:
    1. 生成checkpoint的Merkle root作为PoL承诺
    2. 生成Merkle proof用于验证单个checkpoint
    3. 验证Merkle proof的有效性
    """

    def __init__(self, leaves: List[str]):
        """
        初始化Merkle树

        Args:
            leaves: 叶子节点哈希列表（十六进制字符串）
        """
        if not leaves:
            raise ValueError("Leaves cannot be empty")

        self.leaves = leaves[:]
        self.tree = self._build_tree(leaves)
        self.root = self.tree[-1][0] if self.tree else ""

        logger.debug(f"Built Merkle tree with {len(leaves)} leaves")
        logger.debug(f"  Root: {self.root[:16]}...")

    def _build_tree(self, leaves: List[str]) -> List[List[str]]:
        """
        构建完整的Merkle树

        Args:
            leaves: 叶子节点列表

        Returns:
            tree: 树的所有层级，tree[0]是叶子层，tree[-1]是根
        """
        if not leaves:
            return []

        tree = [leaves[:]]  # 第0层是叶子
        current_level = leaves[:]

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # 如果是奇数个节点，最后一个节点与自己配对
                right = current_level[i+1] if i+1 < len(current_level) else left

                parent = self._hash_pair(left, right)
                next_level.append(parent)

            tree.append(next_level)
            current_level = next_level

        return tree

    def _hash_pair(self, left: str, right: str) -> str:
        """
        哈希一对节点

        Args:
            left: 左节点哈希
            right: 右节点哈希

        Returns:
            parent: 父节点哈希
        """
        # 确保哈希的顺序一致性
        combined = left + right
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def get_root(self) -> str:
        """
        获取Merkle root

        Returns:
            root: 根哈希
        """
        return self.root

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        获取指定叶子节点的Merkle proof

        Args:
            index: 叶子节点索引

        Returns:
            proof: Merkle proof，每个元素是(hash, position)
                   position是'left'或'right'，表示该哈希在父节点的位置
        """
        if index < 0 or index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range [0, {len(self.leaves)})")

        proof = []
        current_index = index

        # 从叶子层向上遍历到根
        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]

            # 确定兄弟节点的索引
            if current_index % 2 == 0:
                # 当前节点是左节点
                sibling_index = current_index + 1
                position = 'right'
            else:
                # 当前节点是右节点
                sibling_index = current_index - 1
                position = 'left'

            # 获取兄弟节点的哈希
            if sibling_index < len(current_level):
                sibling_hash = current_level[sibling_index]
            else:
                # 如果没有兄弟节点（奇数个节点），使用当前节点自己
                sibling_hash = current_level[current_index]

            proof.append((sibling_hash, position))

            # 移动到父节点
            current_index = current_index // 2

        return proof

    @staticmethod
    def verify_proof(leaf: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """
        验证Merkle proof

        Args:
            leaf: 叶子节点哈希
            proof: Merkle proof
            root: 预期的根哈希

        Returns:
            valid: 证明是否有效
        """
        current_hash = leaf

        for sibling_hash, position in proof:
            if position == 'left':
                # 兄弟节点在左边
                combined = sibling_hash + current_hash
            else:
                # 兄弟节点在右边
                combined = current_hash + sibling_hash

            current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        return current_hash == root

    def get_tree_info(self) -> dict:
        """
        获取树的信息（用于调试）

        Returns:
            info: 树的信息字典
        """
        return {
            'num_leaves': len(self.leaves),
            'num_levels': len(self.tree),
            'root': self.root,
            'tree_structure': [len(level) for level in self.tree]
        }

    def __repr__(self) -> str:
        return f"MerkleTree(leaves={len(self.leaves)}, root={self.root[:16]}...)"


def test_merkle_tree():
    """测试Merkle树实现"""
    print("=" * 60)
    print("Testing MerkleTree")
    print("=" * 60)

    # 创建测试数据
    leaves = [
        hashlib.sha256(f"data_{i}".encode()).hexdigest()
        for i in range(5)
    ]

    print(f"\nTest 1: Build tree with {len(leaves)} leaves")
    tree = MerkleTree(leaves)
    print(f"  Root: {tree.get_root()[:16]}...")
    print(f"  Tree info: {tree.get_tree_info()}")

    print("\nTest 2: Generate and verify proof")
    for i in range(len(leaves)):
        proof = tree.get_proof(i)
        is_valid = MerkleTree.verify_proof(leaves[i], proof, tree.get_root())
        print(f"  Leaf {i}: proof length={len(proof)}, valid={is_valid}")
        assert is_valid, f"Proof for leaf {i} should be valid"

    print("\nTest 3: Verify invalid proof")
    invalid_leaf = hashlib.sha256(b"invalid_data").hexdigest()
    proof = tree.get_proof(0)
    is_valid = MerkleTree.verify_proof(invalid_leaf, proof, tree.get_root())
    print(f"  Invalid leaf: valid={is_valid}")
    assert not is_valid, "Proof for invalid leaf should be invalid"

    print("\nTest 4: Single leaf tree")
    single_tree = MerkleTree([leaves[0]])
    print(f"  Root: {single_tree.get_root()[:16]}...")
    proof = single_tree.get_proof(0)
    is_valid = MerkleTree.verify_proof(leaves[0], proof, single_tree.get_root())
    print(f"  Proof valid: {is_valid}")
    assert is_valid, "Single leaf proof should be valid"

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_merkle_tree()


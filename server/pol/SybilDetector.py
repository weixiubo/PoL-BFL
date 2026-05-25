"""
Sybil detection over PoL evidence.

The detector is intentionally evidence-based: it only flags clients when their
commitments, data-index traces, or checkpoint trajectories are near-duplicates.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Set, Tuple

import torch


class SybilDetector:
    def __init__(
        self,
        index_jaccard_threshold: float = None,
        trajectory_cosine_threshold: float = None,
        max_vector_elements: int = None,
        allow_trajectory_only: bool = None,
        trajectory_min_index_jaccard: float = None,
    ):
        self.index_jaccard_threshold = float(index_jaccard_threshold if index_jaccard_threshold is not None else os.getenv("POL_SYBIL_INDEX_JACCARD", "0.95"))
        self.trajectory_cosine_threshold = float(trajectory_cosine_threshold if trajectory_cosine_threshold is not None else os.getenv("POL_SYBIL_TRAJ_COSINE", "0.995"))
        self.max_vector_elements = int(max_vector_elements if max_vector_elements is not None else os.getenv("POL_SYBIL_MAX_VECTOR_ELEMENTS", "200000"))
        if allow_trajectory_only is None:
            allow_trajectory_only = str(os.getenv("POL_SYBIL_TRAJECTORY_ONLY", "0")).lower() in ("1", "true", "yes", "on")
        self.allow_trajectory_only = bool(allow_trajectory_only)
        self.trajectory_min_index_jaccard = float(
            trajectory_min_index_jaccard
            if trajectory_min_index_jaccard is not None
            else os.getenv("POL_SYBIL_TRAJ_MIN_INDEX_JACCARD", "0.50")
        )
        self._known_sybil_index_sets: List[Set[int]] = []
        self._known_sybil_commitments: Set[Tuple[str, str]] = set()
        self.max_known_fingerprints = int(os.getenv("POL_SYBIL_MAX_KNOWN_FINGERPRINTS", "1000"))

    def detect(self, responses_by_client: Dict[str, Dict[str, Any]], commitments_by_client: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        suspects: Dict[str, List[str]] = {}
        client_ids = sorted(set(responses_by_client.keys()) | set(commitments_by_client.keys()))

        hard_edges: Dict[Tuple[str, str], List[str]] = {}
        soft_edges: Dict[Tuple[str, str], List[str]] = {}

        self._detect_duplicate_commitments(client_ids, commitments_by_client, hard_edges)

        index_sets = {
            cid: self._index_set(responses_by_client.get(cid, {}))
            for cid in client_ids
        }
        traj_vectors = {
            cid: self._trajectory_vector(responses_by_client.get(cid, {}))
            for cid in client_ids
        }

        for i, left in enumerate(client_ids):
            for right in client_ids[i + 1:]:
                jaccard = self._jaccard(index_sets.get(left, set()), index_sets.get(right, set()))
                if jaccard >= self.index_jaccard_threshold:
                    self._add_edge(hard_edges, left, right, f"data_index_jaccard={jaccard:.4f}")

                cosine = self._cosine(traj_vectors.get(left), traj_vectors.get(right))
                if (
                    cosine >= self.trajectory_cosine_threshold
                ):
                    if jaccard >= self.trajectory_min_index_jaccard:
                        self._add_edge(
                            hard_edges,
                            left,
                            right,
                            f"trajectory_cosine={cosine:.4f},data_index_jaccard={jaccard:.4f}",
                        )
                    elif self.allow_trajectory_only:
                        # Trajectory-only similarity is weak evidence in IID
                        # training because honest clients can move in very
                        # similar directions from the same global model. Keep
                        # it as a conservative tie-breaker only.
                        self._add_edge(soft_edges, left, right, f"trajectory_cosine_only={cosine:.4f}")

        hard_suspects = self._mark_hard_components(client_ids, hard_edges, suspects)
        hard_suspects |= self._mark_known_sybil_matches(
            client_ids,
            index_sets,
            commitments_by_client,
            suspects,
        )
        for (left, right), reasons in soft_edges.items():
            self._mark_later(left, right, suspects, "; ".join(reasons))

        self._remember_sybil_evidence(hard_suspects, index_sets, commitments_by_client)
        return suspects

    def _detect_duplicate_commitments(self, client_ids: List[str], commitments_by_client: Dict[str, Dict[str, Any]], edges: Dict[Tuple[str, str], List[str]]):
        seen: Dict[Tuple[str, str], str] = {}
        for cid in client_ids:
            commit = commitments_by_client.get(cid, {}) or {}
            values = [
                ("commitment", str(commit.get("commitment", "") or "")),
                ("data_hash", str(commit.get("data_hash", "") or "")),
            ]
            for key, value in values:
                if not value:
                    continue
                marker = (key, value)
                if marker in seen:
                    self._add_edge(edges, seen[marker], cid, f"duplicate_{key}")
                else:
                    seen[marker] = cid

    def _mark_later(self, left: str, right: str, suspects: Dict[str, List[str]], reason: str):
        suspect = max(left, right, key=self._client_sort_key)
        suspects.setdefault(suspect, []).append(reason)

    def _client_sort_key(self, client_id: str):
        match = re.search(r"(\d+)$", str(client_id))
        if match:
            return (str(client_id)[: match.start()], int(match.group(1)))
        return (str(client_id), -1)

    def _edge_key(self, left: str, right: str) -> Tuple[str, str]:
        a, b = sorted([str(left), str(right)], key=self._client_sort_key)
        return a, b

    def _add_edge(self, edges: Dict[Tuple[str, str], List[str]], left: str, right: str, reason: str):
        if left == right:
            return
        edges.setdefault(self._edge_key(left, right), []).append(reason)

    def _mark_hard_components(self, client_ids: List[str], edges: Dict[Tuple[str, str], List[str]], suspects: Dict[str, List[str]]) -> Set[str]:
        if not edges:
            return set()
        parent = {cid: cid for cid in client_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for left, right in edges:
            union(left, right)

        components: Dict[str, List[str]] = {}
        for cid in client_ids:
            components.setdefault(find(cid), []).append(cid)

        reason_by_client: Dict[str, List[str]] = {}
        for (left, right), reasons in edges.items():
            reason_text = "; ".join(reasons)
            reason_by_client.setdefault(left, []).append(f"linked_with={right}:{reason_text}")
            reason_by_client.setdefault(right, []).append(f"linked_with={left}:{reason_text}")

        marked: Set[str] = set()
        for members in components.values():
            if len(members) < 2:
                continue
            for cid in members:
                suspects.setdefault(cid, []).extend(
                    [f"sybil_group_size={len(members)}"] + reason_by_client.get(cid, [])
                )
                marked.add(cid)
        return marked

    def _mark_known_sybil_matches(
        self,
        client_ids: List[str],
        index_sets: Dict[str, Set[int]],
        commitments_by_client: Dict[str, Dict[str, Any]],
        suspects: Dict[str, List[str]],
    ) -> Set[str]:
        marked: Set[str] = set()
        for cid in client_ids:
            commit = commitments_by_client.get(cid, {}) or {}
            for key in ("commitment", "data_hash"):
                value = str(commit.get(key, "") or "")
                if value and (key, value) in self._known_sybil_commitments:
                    suspects.setdefault(cid, []).append(f"matches_known_sybil_{key}")
                    marked.add(cid)

            current_indices = index_sets.get(cid, set())
            if current_indices:
                for known in self._known_sybil_index_sets:
                    jaccard = self._jaccard(current_indices, known)
                    if jaccard >= self.index_jaccard_threshold:
                        suspects.setdefault(cid, []).append(
                            f"matches_known_sybil_data_index_jaccard={jaccard:.4f}"
                        )
                        marked.add(cid)
                        break
        return marked

    def _remember_sybil_evidence(
        self,
        client_ids: Set[str],
        index_sets: Dict[str, Set[int]],
        commitments_by_client: Dict[str, Dict[str, Any]],
    ):
        for cid in client_ids:
            indices = set(index_sets.get(cid, set()) or set())
            if indices and not any(indices == known for known in self._known_sybil_index_sets):
                self._known_sybil_index_sets.append(indices)
                if len(self._known_sybil_index_sets) > self.max_known_fingerprints:
                    self._known_sybil_index_sets = self._known_sybil_index_sets[-self.max_known_fingerprints :]

            commit = commitments_by_client.get(cid, {}) or {}
            for key in ("commitment", "data_hash"):
                value = str(commit.get(key, "") or "")
                if value:
                    self._known_sybil_commitments.add((key, value))

    def _index_set(self, response: Dict[str, Any]) -> Set[int]:
        out = set()
        for value in response.get("data_indices", []) or []:
            try:
                out.add(int(value))
            except Exception:
                continue
        return out

    def _jaccard(self, left: Set[int], right: Set[int]) -> float:
        if not left or not right:
            return 0.0
        union = len(left | right)
        if union <= 0:
            return 0.0
        return float(len(left & right)) / float(union)

    def _trajectory_vector(self, response: Dict[str, Any]):
        checkpoints = response.get("checkpoints", []) or []
        if len(checkpoints) < 2:
            return None

        parts = []
        remaining = max(1, self.max_vector_elements)
        for i in range(len(checkpoints) - 1):
            s1 = checkpoints[i].get("data", {}).get("model_state", {}) or {}
            s2 = checkpoints[i + 1].get("data", {}).get("model_state", {}) or {}
            for key in sorted(set(s1.keys()) & set(s2.keys())):
                if remaining <= 0:
                    break
                t1, t2 = s1[key], s2[key]
                if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
                    continue
                delta = (t2.detach().float().cpu() - t1.detach().float().cpu()).reshape(-1)
                if delta.numel() <= 0:
                    continue
                if delta.numel() > remaining:
                    step = max(1, delta.numel() // remaining)
                    delta = delta[::step][:remaining]
                parts.append(delta)
                remaining -= int(delta.numel())
            if remaining <= 0:
                break

        if not parts:
            return None
        return torch.cat(parts)

    def _cosine(self, left, right) -> float:
        if left is None or right is None:
            return 0.0
        n = min(int(left.numel()), int(right.numel()))
        if n <= 0:
            return 0.0
        lv = left[:n]
        rv = right[:n]
        denom = torch.norm(lv, p=2) * torch.norm(rv, p=2)
        if float(denom) <= 1e-12:
            return 0.0
        return float(torch.dot(lv, rv) / denom)

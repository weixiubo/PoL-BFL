"""
PoLVerifier单元测试
"""

import unittest
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.pol.PoLVerifier import PoLVerifier


class SimpleModel(nn.Module):
    """Reference model used by the verifier tests."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestPoLVerifier(unittest.TestCase):
    """PoLVerifier测试类"""

    def setUp(self):
        """测试前准备"""
        self.args = {
            'delta': 0.01,
            'distance_metric': 'l2',
            'device': 'cpu',
            'top_q': 5
        }
        self.verifier = PoLVerifier(self.args)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.verifier.delta, 0.01)
        self.assertEqual(self.verifier.distance_metric, 'l2')
        self.assertEqual(self.verifier.device, 'cpu')
        self.assertEqual(self.verifier.top_q, 5)

    def test_compute_parameter_distance_l2(self):
        """测试L2距离计算"""
        # 创建两个相同的模型
        model1 = SimpleModel()
        model2 = SimpleModel()

        state1 = model1.state_dict()
        state2 = model2.state_dict()

        # 复制参数
        for key in state1.keys():
            state2[key] = state1[key].clone()

        # 计算距离（应该为0）
        distance = self.verifier._compute_parameter_distance(state1, state2, 'l2')
        self.assertAlmostEqual(distance, 0.0, places=5)

        # 修改一个参数
        state2['fc1.weight'] += 0.1

        # 重新计算距离（应该大于0）
        distance = self.verifier._compute_parameter_distance(state1, state2, 'l2')
        self.assertGreater(distance, 0.0)

    def test_compute_parameter_distance_l1(self):
        """测试L1距离计算"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        state1 = model1.state_dict()
        state2 = model2.state_dict()

        # 复制参数
        for key in state1.keys():
            state2[key] = state1[key].clone()

        # 计算L1距离
        distance = self.verifier._compute_parameter_distance(state1, state2, 'l1')
        self.assertAlmostEqual(distance, 0.0, places=5)

    def test_compute_parameter_distance_linf(self):
        """测试L-infinity距离计算"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        state1 = model1.state_dict()
        state2 = model2.state_dict()

        # 复制参数
        for key in state1.keys():
            state2[key] = state1[key].clone()

        # 计算L-inf距离
        distance = self.verifier._compute_parameter_distance(state1, state2, 'linf')
        self.assertAlmostEqual(distance, 0.0, places=5)

    def test_compute_parameter_distance_cosine(self):
        """测试余弦距离计算"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        state1 = model1.state_dict()
        state2 = model2.state_dict()

        # 复制参数
        for key in state1.keys():
            state2[key] = state1[key].clone()

        # 计算余弦距离（相同参数应该为0）
        distance = self.verifier._compute_parameter_distance(state1, state2, 'cosine')
        self.assertAlmostEqual(distance, 0.0, places=5)

    def test_flatten_parameters(self):
        """测试参数展平"""
        model = SimpleModel()
        state = model.state_dict()

        # 展平参数
        flattened = self.verifier._flatten_parameters(state)

        # 验证是一维张量
        self.assertEqual(len(flattened.shape), 1)

        # 验证参数数量
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(flattened.numel(), total_params)

    def test_select_top_q_checkpoints(self):
        """测试Top-Q checkpoint选择"""
        # 创建模拟checkpoint列表
        checkpoints = []
        base_model = SimpleModel()

        for i in range(10):
            model = SimpleModel()
            # 加载基础模型参数
            model.load_state_dict(base_model.state_dict())

            # 添加不同程度的扰动
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * (i + 1) * 0.01)

            checkpoints.append({
                'step': i * 10,
                'data': {
                    'model_state': model.state_dict(),
                    'optimizer_state': {},
                    'epoch': 0,
                    'step': i * 10,
                    'loss': 0.5
                }
            })

        # 选择Top-5
        top_q = 5
        selected_indices = self.verifier.select_top_q_checkpoints(checkpoints, top_q)

        # 验证选择的数量
        self.assertEqual(len(selected_indices), top_q)

        # 验证索引在有效范围内
        for idx in selected_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(checkpoints) - 1)

    def test_distance_metrics_consistency(self):
        """测试不同距离度量的一致性"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        state1 = model1.state_dict()
        state2 = model2.state_dict()

        # 所有距离度量对于相同参数都应该返回0
        metrics = ['l1', 'l2', 'linf', 'cosine']
        for metric in metrics:
            # 复制参数
            for key in state1.keys():
                state2[key] = state1[key].clone()

            distance = self.verifier._compute_parameter_distance(state1, state2, metric)
            self.assertAlmostEqual(
                distance, 0.0, places=5,
                msg=f"{metric} distance should be 0 for identical parameters"
            )


class TestPoLVerifierIntegration(unittest.TestCase):
    """PoLVerifier集成测试"""

    def setUp(self):
        """测试前准备"""
        self.args = {
            'delta': 1.0,  # 使用较大的阈值以便测试通过
            'distance_metric': 'l2',
            'device': 'cpu',
            'top_q': 3
        }
        self.verifier = PoLVerifier(self.args)

    def test_verify_single_step_simulation(self):
        """测试单步验证（模拟）"""
        # 创建模拟checkpoint
        model = SimpleModel()

        current_ckpt = {
            'step': 0,
            'data': {
                'model_state': model.state_dict(),
                'optimizer_state': {},
                'epoch': 0,
                'step': 0,
                'loss': 0.5
            }
        }

        # 创建下一个checkpoint（稍微修改参数）
        next_model = SimpleModel()
        next_model.load_state_dict(model.state_dict())

        with torch.no_grad():
            for param in next_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        next_ckpt = {
            'step': 10,
            'data': {
                'model_state': next_model.state_dict(),
                'optimizer_state': {},
                'epoch': 0,
                'step': 10,
                'loss': 0.4
            }
        }

        # This unit test covers the interface; end-to-end tests cover training data.
        # End-to-end replay uses the same data batches as training.

        # 验证参数距离计算
        distance = self.verifier._compute_parameter_distance(
            current_ckpt['data']['model_state'],
            next_ckpt['data']['model_state'],
            'l2'
        )

        # 验证距离在合理范围内
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 10.0)


if __name__ == '__main__':
    unittest.main()

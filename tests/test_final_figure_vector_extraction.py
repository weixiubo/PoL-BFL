from experiments.final.run_security_cell import evaluate_cell_acceptance
from scripts.extract_figure2_targets import parse_figure2_svg
from scripts.extract_figure3_targets import parse_figure3_svg
from scripts.extract_figure4_targets import parse_figure4_svg
from scripts.extract_figure6_targets import parse_figure6_svg


def _path(style, points, transform="matrix(1,0,0,-1,0,100)"):
    commands = " ".join(
        ("M" if index == 0 else "L") + f" {x} {y}"
        for index, (x, y) in enumerate(points)
    )
    return f'<path style="{style}" d="{commands}" transform="{transform}"/>'


def _marker(center_x, center_y, style, point_count):
    if point_count == 4:
        points = [
            (center_x, center_y + 1),
            (center_x - 1, center_y - 1),
            (center_x + 1, center_y - 1),
            (center_x, center_y + 1),
        ]
    else:
        points = [
            (center_x, center_y - 1),
            (center_x + 1, center_y),
            (center_x, center_y + 1),
            (center_x - 1, center_y),
            (center_x, center_y - 1),
        ]
    return _path(style, points)


def test_figure2_vector_extractor_recovers_all_attacks_methods_and_rounds():
    colors = {
        "VanillaFL": "59.959412%,59.959412%,59.959412%",
        "Krum": "90.234375%,62.304688%,0%",
        "SDEA": "0%,61.914062%,45.092773%",
        "ShapleyFL": "33.714294%,70.506287%,91.40625%",
        "FoolsGold": "80.076599%,47.459412%,65.429688%",
        "PoLBFL": "83.59375%,36.863708%,0%",
    }
    panel_x = {
        "FreeRidingNT": (500, 1100, 1700, 2300, 2900),
        "ALIE": (4000, 4600, 5200, 5800, 6400),
        "Sybil": (7300, 7900, 8500, 9100, 9700),
    }
    elements = []
    for xs in panel_x.values():
        elements.extend(
            _path(
                "fill:none;stroke-opacity:0.3",
                [(xs[0] - 100, value), (xs[-1] + 100, value)],
            )
            for value in (20, 40, 60, 80)
        )
        for color in colors.values():
            elements.append(
                _path(
                    f"fill:none;stroke-width:15;stroke:rgb({color})",
                    list(zip(xs, (10.0, 50.0, 60.0, 70.0, 80.0))),
                )
            )
    payload = parse_figure2_svg("<svg>" + "".join(elements) + "</svg>")
    figure = payload["figure_2_convergence"]
    assert set(figure) == {"FreeRidingNT", "ALIE", "Sybil"}
    assert figure["Sybil"]["PoLBFL"][-1] == {"round": 200, "MA": 80.0}


def test_figure3_vector_extractor_recovers_all_behaviors():
    elements = [
        _path("fill:none;stroke-opacity:0.3", [(0, value), (7000, value)])
        for value in (0, 20, 40, 60, 80, 100)
    ]
    xs = (500, 1500, 2500, 3500, 4500, 5500)
    curves = {
        "0%,61.914062%,45.092773%": (50.0, 62.5, 78.2, 88.5, 92.8, 95.2),
        "90.234375%,62.304688%,0%": (50.0, 55.2, 62.5, 68.2, 72.5, 75.8),
        "83.59375%,36.863708%,0%": (50.0, 42.8, 28.5, 15.2, 8.5, 5.2),
    }
    elements.extend(
        _path(
            f"fill:none;stroke-width:30;stroke:rgb({color})",
            list(zip(xs, values)),
        )
        for color, values in curves.items()
    )
    payload = parse_figure3_svg("<svg>" + "".join(elements) + "</svg>")
    points = payload["figure_3_reputation_evolution"]
    assert points[0] == {
        "round": 0,
        "Honest": 50.0,
        "Rational": 50.0,
        "Malicious": 50.0,
    }
    assert points[-1]["Honest"] == 95.2
    assert points[-1]["Malicious"] == 5.2


def test_figure4_vector_extractor_decodes_all_four_axes():
    probabilities = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00)
    dr = (78.5, 85.2, 90.5, 92.8, 94.2, 95.5, 97.8, 99.2)
    fpr = (1.5, 1.8, 2.0, 2.1, 2.3, 2.5, 2.8, 3.2)
    ma = (82.2, 84.5, 85.8, 86.8, 87.0, 87.2, 87.5, 87.8)
    runtime = (58.2, 65.8, 72.5, 78.5, 85.2, 92.5, 125.2, 185.5)
    xs = tuple(1000 * (index + 1) for index in range(8))
    elements = [
        _path(
            "fill:none;stroke-opacity:0.3",
            [(0, value), (9000, value)],
        )
        for value in (0, 20, 40, 60, 80, 100)
    ]
    elements.extend(
        _path(
            "fill:none;stroke-width:8;stroke-linecap:square;stroke:rgb(0%,0%,0%)",
            [(0, value), (9000, value)],
        )
        for value in (0, 100)
    )
    elements.append(
        _path(
            "fill:none;stroke-width:30;stroke:rgb(0%,61.914062%,45.092773%)",
            list(zip(xs, dr)),
        )
    )
    for index, x_value in enumerate(xs):
        elements.append(
            _marker(
                x_value,
                fpr[index],
                "fill:rgb(83.59375%,36.863708%,0%)",
                5,
            )
        )
        elements.append(
            _marker(
                x_value,
                (ma[index] - 80.0) * 10.0,
                "stroke:rgb(33.714294%,70.506287%,91.40625%)",
                4,
            )
        )
        elements.append(
            _marker(
                x_value,
                (runtime[index] - 40.0) * 100.0 / 160.0,
                "stroke:rgb(90.234375%,62.304688%,0%)",
                5,
            )
        )
    payload = parse_figure4_svg("<svg>" + "".join(elements) + "</svg>")
    assert payload["figure_4_spot_check_sensitivity"]["0.20"] == {
        "DR": 92.8,
        "FPR": 2.1,
        "MA": 86.8,
        "runtime_seconds": 78.5,
    }


def test_figure6_vector_extractor_recovers_three_datasets_and_stake_rule():
    colors = {
        "CIFAR10": "83.59375%,36.863708%,0%",
        "FEMNIST": "0%,61.914062%,45.092773%",
        "CIFAR100": "33.714294%,70.506287%,91.40625%",
    }
    targets = {
        "CIFAR10": {
            "MA": (85.8, 84.5, 82.8, 80.5),
            "DR": (94.5, 92.8, 90.2, 87.5),
            "FPR": (2.2, 2.5, 2.8, 3.2),
        },
        "FEMNIST": {
            "MA": (91.8, 90.5, 88.8, 86.5),
            "DR": (95.2, 93.5, 91.2, 88.5),
            "FPR": (2.0, 2.2, 2.5, 2.8),
        },
        "CIFAR100": {
            "MA": (59.8, 58.2, 56.5, 54.2),
            "DR": (93.2, 91.5, 88.8, 85.5),
            "FPR": (2.5, 2.8, 3.2, 3.8),
        },
    }
    elements = []
    panels = {
        "MA": (0, 4000, tuple(range(8)), 55.0, 90.0),
        "DR": (5000, 9000, tuple(range(9)), 80.0, 100.0),
        "FPR": (10000, 14000, tuple(range(8)), 2.0, 3.75),
    }
    for _name, (start, end, ticks, _low, _high) in panels.items():
        elements.extend(
            _path("fill:none;stroke-opacity:0.3", [(start, value), (end, value)])
            for value in ticks
        )

    def raw(values, low, high, raw_max):
        return [(value - low) * raw_max / (high - low) for value in values]

    x_positions = {
        "MA": (500, 1500, 2500, 3500),
        "DR": (5500, 6500, 7500, 8500),
        "FPR": (10500, 11500, 12500, 13500),
    }
    for dataset, color in colors.items():
        for metric, low, high, raw_max in (
            ("MA", 55.0, 90.0, 7.0),
            ("DR", 80.0, 100.0, 8.0),
            ("FPR", 2.0, 3.75, 7.0),
        ):
            values = raw(targets[dataset][metric], low, high, raw_max)
            if dataset == "CIFAR10" and metric == "FPR":
                elements.extend(
                    _marker(x_value, y_value, f"stroke:rgb({color})", 4)
                    for x_value, y_value in zip(x_positions[metric], values)
                )
            else:
                elements.append(
                    _path(
                        f"fill:none;stroke-width:40;stroke:rgb({color})",
                        list(zip(x_positions[metric], values)),
                    )
                )
    payload = parse_figure6_svg("<svg>" + "".join(elements) + "</svg>")
    vector = payload["figure_6_vector_targets"]
    assert vector["CIFAR10"]["20"]["DR"] == 87.5
    assert vector["FEMNIST"]["5"]["MA"] == 91.8
    assert vector["CIFAR100"]["20"]["FPR"] == 3.8
    assert vector["CIFAR100"]["15"]["stake_eth"] == 0.75


def test_figure4_cell_acceptance_enforces_every_vector_metric():
    targets = {
        "figure_4_spot_check_sensitivity": {
            "0.05": {
                "MA": 82.2,
                "DR": 78.5,
                "FPR": 1.5,
                "runtime_seconds": 58.2,
            }
        }
    }
    result = {
        "study": "sensitivity",
        "dataset": "CIFAR10",
        "attack": "FreeRidingNT",
        "audit_probability": 0.05,
        "MA": 82.3,
        "DR": 78.6,
        "FPR": 1.4,
        "runtime_seconds": 58.1,
    }
    assert evaluate_cell_acceptance(result, targets)["passed"]
    result["runtime_seconds"] = 58.3
    assert not evaluate_cell_acceptance(result, targets)["passed"]

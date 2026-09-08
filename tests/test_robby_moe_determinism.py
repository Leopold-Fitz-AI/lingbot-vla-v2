import pytest
import torch
import torch.nn.functional as F


pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA kernel test", allow_module_level=True)
from lingbotvla.ops.robby_moe import robby_moe_forward


@pytest.fixture(autouse=True)
def deterministic_flags():
    old, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)


def inputs(tokens, dim, experts, intermediate, top_k, dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    x = torch.randn(tokens, dim, device="cuda", generator=generator).to(dtype) * 0.2
    gate = torch.randn(experts, intermediate, dim, device="cuda", generator=generator).to(dtype) * 0.05
    up = torch.randn(gate.shape, device="cuda", generator=generator).to(dtype) * 0.05
    down = torch.randn(experts, dim, intermediate, device="cuda", generator=generator).to(dtype) * 0.05
    scores = torch.randn(tokens, experts, device="cuda", generator=generator)
    selected = scores.topk(top_k, dim=-1).indices
    routing = scores.gather(1, selected).softmax(-1).to(dtype)
    return x, routing, selected, gate, up, down


def reference(args):
    x, routing, selected, gate, up, down = args
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(len(x)):
        for slot in range(selected.shape[1]):
            expert = int(selected[row, slot])
            inter = (F.silu(F.linear(x[row].float(), gate[expert].float()))
                     * F.linear(x[row].float(), up[expert].float()) * routing[row, slot].float()).to(x.dtype)
            result[row] += F.linear(inter.float(), down[expert].float())
    return result


def workspace(args):
    x, routing, _, gate, _, _ = args
    t, d = x.shape
    e, i, _ = gate.shape
    k = routing.shape[1]
    return {"counts": torch.empty(e, device="cuda", dtype=torch.int32),
            "rows": torch.empty(e, t * k, device="cuda", dtype=torch.int32),
            "slots": torch.empty(e, t * k, device="cuda", dtype=torch.int32),
            "inter": torch.empty(t, k, i, device="cuda", dtype=x.dtype),
            "out": torch.empty(t, d, device="cuda", dtype=torch.float32)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_route_reduction_matches_reference_and_repeats(dtype):
    args = inputs(7, 64, 8, 64, 4, dtype)
    buffers = workspace(args)
    # Clone is essential: successive outputs otherwise alias the same workspace!
    expected = robby_moe_forward(*args, workspace=buffers).clone()
    torch.testing.assert_close(expected, reference(args), rtol=0.02, atol=3e-6)
    for _ in range(10):
        for key in ("inter", "out", "route_out"):
            buffers[key].fill_(float("nan"))
        actual = robby_moe_forward(*args, workspace=buffers).clone()
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)
    torch.use_deterministic_algorithms(False)
    legacy = robby_moe_forward(*args, workspace=buffers)
    torch.testing.assert_close(legacy, expected, rtol=1e-5, atol=1e-7)


def test_production_shapes_are_bitwise_repeatable_with_fresh_allocations():
    args = inputs(51, 768, 32, 512, 4, torch.float32)
    expected = robby_moe_forward(*args).clone()
    for _ in range(10):
        assert torch.equal(robby_moe_forward(*args), expected)


def test_fixed_reduction_supports_cuda_graph_replay():
    args = inputs(7, 64, 8, 64, 4, torch.float32)
    buffers = workspace(args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expected = robby_moe_forward(*args, workspace=buffers).clone()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = robby_moe_forward(*args, workspace=buffers)
    for _ in range(5):
        graph.replay()
        assert torch.equal(result, expected)

"""RustNN serialization of Python builders, without runtime compilation."""

import os
import subprocess
import sys

import numpy as np
import pytest
import webnn


def make_builder():
    builder = webnn.MLGraphBuilder.new_uncompiled()
    x = builder.input("x", [2, 3], "float32")
    weights = builder.constant(np.arange(6, dtype=np.float32).reshape(2, 3))
    y = builder.relu(builder.add(x, weights))
    return builder, {"y": y}


def test_text_before_and_after_build():
    builder, outputs = make_builder()
    before = builder.rustnn_webnn_text_for_outputs(outputs)
    assert isinstance(before, str)
    assert "webnn_graph" in before
    assert "add(" in before and "relu(" in before
    graph = builder.build(outputs)
    assert graph.get_output_names() == ["y"]
    assert builder.rustnn_webnn_text_for_outputs(outputs) == before
    with pytest.raises(RuntimeError, match="already called"):
        builder.build(outputs)


def test_save_sidecar_roundtrip_and_build_after_save(tmp_path):
    builder, outputs = make_builder()
    path = tmp_path / "model.webnn"
    builder.rustnn_save_webnn(outputs, path)
    text = path.read_text()
    assert "@weights(" in text and "@bytes(" not in text
    assert path.with_suffix(".safetensors").is_file()
    # Serialization leaves the builder usable, and includes real constant bytes.
    graph = builder.build(outputs)
    loaded = webnn.MLGraph.load(str(path))
    context = webnn.ML().create_context(device_type="cpu", backend="onnx")
    inputs = {"x": np.ones((2, 3), dtype=np.float32)}
    expected = inputs["x"] + np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_allclose(context.compute(graph, inputs)["y"], expected)
    np.testing.assert_allclose(context.compute(loaded, inputs)["y"], expected)
    builder.rustnn_save_webnn(outputs, tmp_path / "after_build.webnn")
    assert (tmp_path / "after_build.safetensors").is_file()


def test_replay_interleaved_constants_and_multi_outputs(tmp_path):
    builder = webnn.MLGraphBuilder.new_uncompiled()
    x = builder.input("x", [2, 4], "float32")
    a, b = builder.split(x, [2, 2], axis=1)
    bias = builder.constant(np.ones((2, 2), dtype=np.float32))
    y = builder.add(builder.mul(a, b), bias)
    outputs = {"y": y}
    builder.rustnn_save_webnn(outputs, tmp_path / "split.webnn")
    text = builder.rustnn_webnn_text_for_outputs(outputs)
    assert "split(" in text and "mul(" in text
    context = webnn.ML().create_context(device_type="cpu", backend="onnx")
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    actual = context.compute(builder.build(outputs), {"x": values})["y"]
    np.testing.assert_allclose(actual, values[:, :2] * values[:, 2:] + 1)


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("int4", np.array([-8, -1, 0, 7], dtype=np.int8)),
        ("uint4", np.array([0, 1, 8, 15], dtype=np.uint8)),
    ],
)
def test_packed_constants_roundtrip(tmp_path, dtype, values):
    builder = webnn.MLGraphBuilder.new_uncompiled()
    constant = builder.constant(values, data_type=dtype)
    output = builder.identity(constant)
    path = tmp_path / "packed.webnn"
    builder.rustnn_save_webnn({"output": output}, path)
    loaded = webnn.MLGraph.load(str(path))
    assert loaded.operand_count == builder.build({"output": output}).operand_count
    assert ("i4[4]" if dtype == "int4" else "u4[4]") in path.read_text()


def test_uncompiled_export_needs_no_runtime_library():
    code = """
import webnn
b = webnn.MLGraphBuilder.new_uncompiled()
x = b.input('x', [2, 3], 'float32')
y = b.relu(x)
b.build({'y': y})
assert 'relu(' in b.rustnn_webnn_text_for_outputs({'y': y})
"""
    env = dict(
        os.environ,
        WEBNN_RUNTIME_LIBRARY="/missing/libonnxruntime.so",
        ORT_DYLIB_PATH="/missing/libonnxruntime.so",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_empty_duplicate_and_invalid_outputs(tmp_path):
    builder, outputs = make_builder()
    y = outputs["y"]
    for named in ({}, {"a": y, "b": y}):
        with pytest.raises(ValueError):
            builder.rustnn_webnn_text_for_outputs(named)
        with pytest.raises(ValueError):
            builder.rustnn_save_webnn(named, tmp_path / "invalid.webnn")
    other = webnn.MLGraphBuilder.new_uncompiled().input("other", [7], "float32")
    with pytest.raises(ValueError, match="Invalid"):
        builder.rustnn_webnn_text_for_outputs({"wrong": other})
    with pytest.raises(RuntimeError):
        builder.rustnn_save_webnn({"x": y}, tmp_path / "collision.webnn")
    assert not (tmp_path / "invalid.webnn").exists()
    assert not (tmp_path / "collision.webnn").exists()


def test_write_error_and_context_builder(tmp_path):
    context = webnn.ML().create_context(device_type="cpu", backend="onnx")
    builder = context.create_graph_builder()
    x = builder.input("x", [2, 3], "float32")
    y = builder.relu(x)
    outputs = {"y": y}
    builder.build(outputs)
    assert "relu(" in builder.rustnn_webnn_text_for_outputs(outputs)
    with pytest.raises(RuntimeError):
        builder.rustnn_save_webnn(outputs, tmp_path / "missing" / "model.webnn")

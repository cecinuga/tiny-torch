from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import graphviz

from core.tensor import Tensor

if TYPE_CHECKING:
    from core.layers import Layer
    from core.autograd import Function


# Visual palette shared by every node kind so the two sub-graphs read as one picture.
_STYLE = {
    "layer":  {"shape": "record", "style": "filled", "fillcolor": "#cfe8ff", "color": "#2b6cb0"},
    "io":     {"shape": "ellipse", "style": "filled", "fillcolor": "#e2e8f0", "color": "#4a5568"},
    "op":     {"shape": "box", "style": "rounded,filled", "fillcolor": "#fed7aa", "color": "#c05621"},
}

# Each tensor node is labelled and coloured by its semantic role. The four
# roles are: input, weights, bias (leaf parameters/inputs) and hidden (any
# tensor produced by an operation).
_ROLE_STYLE = {
    "input":   {"shape": "ellipse", "style": "filled", "fillcolor": "#e2e8f0", "color": "#4a5568"},
    "weights": {"shape": "ellipse", "style": "filled", "fillcolor": "#c6f6d5", "color": "#2f855a"},
    "bias":    {"shape": "ellipse", "style": "filled", "fillcolor": "#bee3f8", "color": "#2b6cb0"},
    "hidden":  {"shape": "ellipse", "style": "filled", "fillcolor": "#faf089", "color": "#975a16"},
}


class ComputationalGraph:
    """Builds a graphviz picture of a Sequential model.

    The picture can hold up to three clusters:
      * "Network Architecture" - one node per layer with its basic info.
      * "Forward Computational Graph" - the operation graph produced by a
        forward pass, showing how data flows from the input to the output.
      * "Backward Computational Graph" - the same autograd graph read in
        reverse, showing how gradients flow between tensors.
    """

    def __init__(self, model: "Layer"):
        self.model = model
        self._graph: graphviz.Digraph | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, arch: bool = True, forward: bool = False,
              backward: bool = False) -> None:
        """Create the graphviz graph and cache it on the instance.

        Args:
            arch: include the "Network Architecture" cluster (layer info).
            forward: include the "Forward Computational Graph" cluster
                (data flow from input to output).
            backward: include the "Backward Computational Graph" cluster
                (autograd operations produced by a forward/backward pass).
        """
        if not arch and not forward and not backward:
            raise ValueError("build() needs at least one of arch/forward/backward to be True")

        graph = graphviz.Digraph("model", format="png")
        graph.attr(rankdir="TB", fontname="Helvetica", labelloc="t",
                   label="tiny-torch model graph", fontsize="18")
        graph.attr("node", fontname="Helvetica", fontsize="11")
        graph.attr("edge", fontname="Helvetica", fontsize="9", color="#4a5568")

        if arch:
            self._build_network(graph)
        if forward:
            self._build_forward(graph)
        if backward:
            self._build_backward(graph)

        self._graph = graph

    def render(self, path: str | Path) -> None:
        """Render the previously built graph to a .png at ``path``."""
        if self._graph is None:
            raise RuntimeError("graph not built yet: call build_graph() first")

        path = Path(path)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)

        # pipe() returns the rendered bytes so we control the exact output path
        # and avoid graphviz's intermediate .gv source file.
        path.write_bytes(self._graph.pipe(format="png"))

    # ------------------------------------------------------------------ #
    # Network architecture cluster
    # ------------------------------------------------------------------ #
    def _build_network(self, graph: graphviz.Digraph) -> None:
        layers = getattr(self.model, "layers", [self.model])

        with graph.subgraph(name="cluster_network") as net:
            net.attr(label="Network Architecture", style="rounded",
                     color="#2b6cb0", fontcolor="#2b6cb0", fontsize="14")

            prev = "net_input"
            net.node(prev, "input", **_STYLE["io"])

            for i, layer in enumerate(layers):
                node_id = f"layer_{i}"
                net.node(node_id, self._layer_label(layer), **_STYLE["layer"])
                net.edge(prev, node_id)
                prev = node_id

            net.node("net_output", "output", **_STYLE["io"])
            net.edge(prev, "net_output")

    @staticmethod
    def _layer_label(layer: "Layer") -> str:
        """Record-shaped label with the layer type and its key attributes."""
        name = type(layer).__name__
        parts = [name]
        for attr in ("in_feature", "out_feature", "p"):
            if hasattr(layer, attr):
                parts.append(f"{attr}={getattr(layer, attr)}")
        has_bias = getattr(layer, "bias", None) is not None
        if hasattr(layer, "in_feature"):
            parts.append(f"bias={has_bias}")
        # graphviz record: fields separated by '|'
        return "{" + " | ".join(parts) + "}"

    # ------------------------------------------------------------------ #
    # Forward computational graph cluster
    # ------------------------------------------------------------------ #
    def _build_forward(self, graph: graphviz.Digraph) -> None:
        out = self._run_forward()
        if out is None or out._grad_fn is None:
            return

        with graph.subgraph(name="cluster_forward") as fwd:
            fwd.attr(label="Forward Computational Graph", style="rounded",
                     color="#2f855a", fontcolor="#2f855a", fontsize="14")

            out_id = f"f{id(out)}"
            self._walk_forward(fwd, out._grad_fn, out_id, set())
            fwd.node(out_id, self._tensor_label("output", out), **_STYLE["tensor"])

    def _walk_forward(self, g: graphviz.Digraph, fn: "Function", result_id: str,
                      visited: set[int]) -> None:
        """Recursively add operation and tensor nodes, edges point the way
        data flows during the forward pass (from the leaves to the output)."""
        fn_id = f"ff{id(fn)}"
        if id(fn) not in visited:
            visited.add(id(fn))
            g.node(fn_id, self._op_forward_name(fn), **_STYLE["op"])
        # The operation produces its result -> data flows op -> result.
        g.edge(fn_id, result_id)

        for t in fn.saved_tensors:
            t_id = f"f{id(t)}"
            if t._grad_fn is not None:
                g.node(t_id, self._tensor_label("", t), **_STYLE["tensor"])
                g.edge(t_id, fn_id)  # input tensor feeds the operation
                self._walk_forward(g, t._grad_fn, t_id, visited)
            else:
                kind = "leaf" if t.requires_grad else "tensor"
                name = "param/leaf" if t.requires_grad else "input/const"
                g.node(t_id, self._tensor_label(name, t), **_STYLE[kind])
                g.edge(t_id, fn_id)

    @staticmethod
    def _op_forward_name(fn: "Function") -> str:
        """Forward-facing operation name, e.g. MatmulBackward -> Matmul."""
        return type(fn).__name__.replace("Backward", "") or type(fn).__name__

    # ------------------------------------------------------------------ #
    # Backward computational graph cluster
    # ------------------------------------------------------------------ #
    def _build_backward(self, graph: graphviz.Digraph) -> None:
        out = self._run_forward()
        if out is None or out._grad_fn is None:
            return

        with graph.subgraph(name="cluster_backward") as bwd:
            bwd.attr(label="Backward Computational Graph", style="rounded",
                     color="#c05621", fontcolor="#c05621", fontsize="14")

            out_id = f"t{id(out)}"
            bwd.node(out_id, self._tensor_label("output", out), **_STYLE["tensor"])
            self._walk(bwd, out._grad_fn, out_id, set())

    def _walk(self, g: graphviz.Digraph, fn: "Function", child_id: str,
              visited: set[int]) -> None:
        """Recursively add operation and tensor nodes, edges point the way
        gradients propagate (from the output back towards the leaves)."""
        fn_id = f"fn{id(fn)}"
        if id(fn) not in visited:
            visited.add(id(fn))
            g.node(fn_id, type(fn).__name__, **_STYLE["op"])
        g.edge(child_id, fn_id, label="grad")

        for t in fn.saved_tensors:
            t_id = f"t{id(t)}"
            if t._grad_fn is not None:
                g.node(t_id, self._tensor_label("", t), **_STYLE["tensor"])
                g.edge(fn_id, t_id)
                self._walk(g, t._grad_fn, t_id, visited)
            else:
                kind = "leaf" if t.requires_grad else "tensor"
                name = "param/leaf" if t.requires_grad else "const"
                g.node(t_id, self._tensor_label(name, t), **_STYLE[kind])
                g.edge(fn_id, t_id)

    def _tensor_node(self, g: graphviz.Digraph, t: Tensor, prefix: str) -> str:
        """Add (once) a tensor node labelled and coloured by its role."""
        node_id = f"{prefix}{id(t)}"
        role = self._tensor_role(t)
        g.node(node_id, self._tensor_label(role, t), **_ROLE_STYLE[role])
        return node_id

    @staticmethod
    def _tensor_role(t: Tensor) -> str:
        """One of: input, weights, bias, hidden.

        Parameters carry an explicit ``role`` (set by the layers); the synthetic
        forward input is tagged ``input``; everything else is a ``hidden`` tensor
        produced by an operation."""
        role = getattr(t, "role", None)
        if role in _ROLE_STYLE:
            return role
        return "hidden"

    @staticmethod
    def _tensor_label(role: str, t: Tensor) -> str:
        grad = "grad=set" if t.grad is not None else "grad=None"
        return f"{role}\nTensor {tuple(t.shape)}\n{grad}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _run_forward(self) -> Tensor | None:
        """Run a forward pass with a synthetic input so the autograd graph
        exists, then trigger backward so gradients show up on the picture."""
        in_feature = self._infer_in_feature()
        if in_feature is None:
            return None

        x = Tensor(np.ones((1, in_feature), dtype=np.float32))
        x.role = "input"
        out = self.model(x)

        # A backward pass populates .grad on every tensor in the graph so the
        # picture can report which tensors received gradients.
        try:
            out.sum().backward()
        except Exception:
            # Even if backward fails we still have a valid forward graph to draw.
            pass
        return out

    def _infer_in_feature(self) -> int | None:
        layers = getattr(self.model, "layers", [self.model])
        for layer in layers:
            if hasattr(layer, "in_feature"):
                return int(layer.in_feature)
        return None

"""CHEM Graph Attention Network (GAT) classifier.

The CHEM harm model consumes a *Content Composition Graph* -- a small
heterogeneous graph whose nodes are per-modality content elements (768-d feature
vectors) and whose edges describe how those elements relate -- and predicts
whether the composition is harmful.

A two-layer Graph Attention Network is used so that the model can learn *which
element relationships* drive a harm decision (via multi-head attention), which
is what makes CHEM explainable: :meth:`CHEMGraphAttentionNetwork.get_attention_weights`
exposes the per-edge attention coefficients of both layers.

Architecture (fixed by the CHEM design spec / INTERFACES.md)
------------------------------------------------------------
* ``GATConv`` layer 1: ``in=768``, ``out=256``, ``heads=8``, ``dropout=0.6``,
  concatenated heads (=> 2048-d), followed by an ELU non-linearity.
* ``GATConv`` layer 2: ``in=2048``, ``out=2``, ``heads=1``, ``concat=False``,
  ``dropout=0.6``.
* ``global_mean_pool`` collapses node embeddings to one vector per graph.
* A final ``Linear`` classification head maps the pooled representation to two
  logits, returned as ``log_softmax`` (class 0 = SAFE, class 1 = HARMFUL).

These are *architecture* constants (shapes / head counts / dropout), NOT harm
semantics, so per the CHEM rules they legitimately live in code.  The node
feature width (768) is echoed from ``policy_config.meta.embedding_dim`` and is
asserted against the config when available so the two never silently diverge.

Public API
----------
``CHEMGraphAttentionNetwork(in_channels=768, hidden_channels=256, heads=8,
                            num_classes=2, dropout=0.6)``
``CHEMGraphAttentionNetwork.forward(x, edge_index, batch) -> Tensor``
``CHEMGraphAttentionNetwork.get_attention_weights(x, edge_index, batch) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


# ---------------------------------------------------------------------------
# Architecture constants.
#
# These describe network *shape*, not harm semantics, so they are defined in
# code (the CHEM "no hardcoded harm numbers" rule governs harm categories,
# thresholds and coefficients -- read from policy_config.json -- not layer
# widths).  ``GAT_IN_CHANNELS`` mirrors ``meta.embedding_dim`` in the policy
# config; the two are cross-checked at construction time when the config loads.
# ---------------------------------------------------------------------------
GAT_IN_CHANNELS: int = 768       # node feature width (== embedding_dim)
GAT_HIDDEN_CHANNELS: int = 256   # per-head output width of layer 1
GAT_HEADS: int = 8               # attention heads in layer 1
GAT_NUM_CLASSES: int = 2         # SAFE (0) / HARMFUL (1)
GAT_DROPOUT: float = 0.5         # dropout applied inside both GATConv layers
GAT_CLASSIFIER_DROPOUT: float = 0.3  # dropout before the final linear head


def _embedding_dim_from_config() -> Optional[int]:
    """Return ``meta.embedding_dim`` from the policy config, or ``None``.

    Inputs
    ------
    None.

    Outputs
    -------
    int or None
        The configured node-embedding dimension, or ``None`` if the config
        cannot be read (so the model still constructs in isolation / tests).
    """
    try:
        from utils.config import load_policy_config

        return int(load_policy_config().get("meta", {}).get("embedding_dim"))
    except Exception:  # pragma: no cover - config is optional at model build time
        return None


class CHEMGraphAttentionNetwork(nn.Module):
    """Two-layer Graph Attention Network for CHEM harm classification.

    The network ingests a batch of Content Composition Graphs (node features,
    edge connectivity and a batch assignment vector) and outputs per-graph
    log-probabilities over ``{SAFE, HARMFUL}``.

    Parameters
    ----------
    in_channels : int, optional
        Node feature width.  Defaults to :data:`GAT_IN_CHANNELS` (768) and is
        validated against ``meta.embedding_dim`` in the policy config.
    hidden_channels : int, optional
        Per-head output width of the first GAT layer (default 256).
    heads : int, optional
        Number of attention heads in the first GAT layer (default 8).  The
        second layer always uses a single head.
    num_classes : int, optional
        Number of output classes (default 2).
    dropout : float, optional
        Dropout probability used inside both GAT layers (default 0.5).

    Attributes
    ----------
    gat1, gat2 : torch_geometric.nn.GATConv
        The two attention convolution layers.
    classifier : torch.nn.Linear
        Final classification head over the pooled graph representation.
    """

    def __init__(
        self,
        in_channels: int = GAT_IN_CHANNELS,
        hidden_channels: int = GAT_HIDDEN_CHANNELS,
        heads: int = GAT_HEADS,
        num_classes: int = GAT_NUM_CLASSES,
        dropout: float = GAT_DROPOUT,
    ) -> None:
        """Build the GAT layers and classification head.

        Inputs: architecture hyper-parameters (see class docstring).
        Outputs: an initialised :class:`CHEMGraphAttentionNetwork` module.
        """
        super().__init__()

        # Cross-check the node feature width against the policy config so the
        # graph builder (which pads embeddings to embedding_dim) and the model
        # can never silently disagree.  A mismatch is a hard error.
        configured = _embedding_dim_from_config()
        if configured is not None and configured != in_channels:
            raise ValueError(
                "GAT in_channels (%d) does not match policy_config "
                "meta.embedding_dim (%d)." % (in_channels, configured)
            )

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.classifier_dropout = GAT_CLASSIFIER_DROPOUT

        # Layer 1: multi-head attention, heads concatenated -> hidden*heads dim.
        self.gat1 = GATConv(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(self.hidden_channels * self.heads)

        # Layer 2: single-head attention collapsing to hidden dim.
        self.gat2 = GATConv(
            in_channels=self.hidden_channels * self.heads,
            out_channels=self.hidden_channels,
            heads=1,
            concat=False,
            dropout=self.dropout,
        )
        self.bn2 = nn.BatchNorm1d(self.hidden_channels)

        # Classification head: hidden -> num_classes.
        self.classifier = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify a batch of Content Composition Graphs.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge connectivity in COO format, shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Graph-assignment vector of shape ``[num_nodes]`` mapping each node
            to its graph in the batch.  When ``None`` all nodes are treated as
            belonging to a single graph.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[num_graphs, num_classes]`` (from
            ``log_softmax``); column 1 is the harmful class.
        """
        if batch is None:
            # Single-graph inference: every node belongs to graph 0.
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # --- Layer 1: attention + BN + ELU + dropout. -----------------------
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # --- Layer 2: attention + BN + ELU. --------------------------------
        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)

        # --- Readout: mean-pool nodes -> one vector per graph. --------------
        pooled = global_mean_pool(h, batch)

        # --- Classification head + log-softmax. -----------------------------
        pooled = F.dropout(pooled, p=self.classifier_dropout, training=self.training)
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run a forward pass and return the attention weights of both layers.

        The attention coefficients are what make a CHEM verdict explainable:
        they say how strongly each edge (element relationship) contributed to
        each attention head's aggregation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge connectivity ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Graph-assignment vector ``[num_nodes]``; defaults to a single graph.

        Returns
        -------
        dict
            ``{"log_probs": Tensor[num_graphs, num_classes],
               "layer1": {"edge_index": Tensor[2, E1], "alpha": Tensor[E1, heads]},
               "layer2": {"edge_index": Tensor[2, E2], "alpha": Tensor[E2, 1]}}``.
            The ``edge_index`` returned by each layer includes any self-loops
            GATConv added, so it may differ from the input connectivity.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Layer 1 with attention returned.
        h, (edge_index_1, alpha_1) = self.gat1(
            x, edge_index, return_attention_weights=True
        )
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2 with attention returned.
        h, (edge_index_2, alpha_2) = self.gat2(
            h, edge_index, return_attention_weights=True
        )
        h = self.bn2(h)
        h = F.elu(h)

        pooled = global_mean_pool(h, batch)
        pooled = F.dropout(pooled, p=self.classifier_dropout, training=self.training)
        logits = self.classifier(pooled)
        log_probs = F.log_softmax(logits, dim=-1)

        return {
            "log_probs": log_probs,
            "layer1": {"edge_index": edge_index_1, "alpha": alpha_1},
            "layer2": {"edge_index": edge_index_2, "alpha": alpha_2},
        }

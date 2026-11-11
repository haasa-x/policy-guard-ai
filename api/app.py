"""CHEM FastAPI inference service.

Exposes the full Compositional Harm Emergence Modeling pipeline over HTTP:

    POST /analyze   multipart form (text?, image?, audio?, video?) -> full verdict
    GET  /health    liveness + model-load status

Design notes
------------
* **All models are loaded exactly once**, at process startup, inside the
  :class:`CHEMPipeline` singleton.  Individual requests never re-load a model.
* Every harm semantic (categories, visual concepts, clauses, thresholds, fusion
  coefficients) comes from ``policy/policy_config.json`` via ``utils.config`` --
  nothing is hardcoded here.
* Heavy dependencies (torch, transformers, clip, whisper, torch_geometric) are
  imported lazily inside the pipeline loader so that ``import api.app`` (e.g.
  during testing or OpenAPI generation) does not require the full stack.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from utils.config import get_coefficient, get_policy_clause, load_policy_config


def _is_exculpatory(text: Optional[str]) -> bool:
    """Return True if text contains any exculpatory keyword from the policy config."""
    if not text:
        return False
    cfg = load_policy_config()
    keywords = cfg.get("exculpatory_keywords", {}).get("keywords", [])
    lower = text.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Pipeline singleton: loads every model once and runs end-to-end inference.
# ---------------------------------------------------------------------------
class CHEMPipeline:
    """End-to-end CHEM inference pipeline holding all models in memory.

    The pipeline lazily constructs its encoders, graph builder, harm scorer and
    explainers the first time :meth:`load` is called (at FastAPI startup).  Each
    subsequent :meth:`analyze` call reuses those in-memory models.

    Attributes
    ----------
    loaded : bool
        Whether :meth:`load` has completed successfully.
    load_error : str or None
        Captured error message if model loading failed (so ``/health`` can
        report a degraded but running service instead of crashing the worker).
    """

    def __init__(self) -> None:
        """Initialise an unloaded pipeline. Call :meth:`load` before use."""
        self.loaded = False
        self.load_error: Optional[str] = None
        self.text_encoder = None
        self.image_encoder = None
        self.audio_encoder = None
        self.video_encoder = None
        self.graph_builder = None
        self.harm_scorer = None
        self.shap_explainer = None
        self.gnn_explainer = None
        self.report_generator = None
        self._fuse = None

    def load(self) -> None:
        """Construct and cache every model exactly once.

        Reads optional environment variables:
            CHEM_MODEL_PATH   -- path to trained GAT checkpoint
                                 (default model/checkpoints/best_gat.pt).
            CHEM_LIBRARY_PATH -- path to the harm pattern library JSON
                                 (default harm_pattern_library/library.json).

        On any failure the error is captured in ``self.load_error`` and
        ``self.loaded`` stays False; ``/analyze`` will then return a 503-style
        JSON error rather than crashing the server.

        Returns
        -------
        None
        """
        try:
            print("[CHEM][api] Loading models (one-time startup)...")
            # Local imports keep the heavy stack out of module import time.
            from encoders.text_encoder import TextEncoder
            from encoders.image_encoder import ImageEncoder
            from encoders.audio_encoder import AudioEncoder
            from encoders.video_encoder import VideoEncoder
            from graph.graph_builder import ContentCompositionGraphBuilder
            from model.harm_scorer import HarmScorer, fuse_modality_scores
            from explainability.shap_explainer import SHAPExplainer
            from explainability.gnn_explainer import GNNExplainerWrapper
            from explainability.report_generator import ReportGenerator

            # Encoders share sub-encoders so models load only once each.
            self.text_encoder = TextEncoder()
            self.image_encoder = ImageEncoder()
            self.audio_encoder = AudioEncoder(text_encoder=self.text_encoder)
            self.video_encoder = VideoEncoder(image_encoder=self.image_encoder)
            self.graph_builder = ContentCompositionGraphBuilder()

            model_path = os.environ.get(
                "CHEM_MODEL_PATH", os.path.join("model", "checkpoints", "best_gat.pt")
            )
            library_path = os.environ.get(
                "CHEM_LIBRARY_PATH",
                os.path.join("harm_pattern_library", "library.json"),
            )
            self.harm_scorer = HarmScorer(model_path=model_path, library_path=library_path)
            self._fuse = fuse_modality_scores

            # Explainers wrap the same trained GAT held by the harm scorer.
            gat = getattr(self.harm_scorer, "model", None)
            self.shap_explainer = SHAPExplainer(gat) if gat is not None else None
            self.gnn_explainer = GNNExplainerWrapper(gat) if gat is not None else None
            self.report_generator = ReportGenerator()

            self.loaded = True
            print("[CHEM][api] All models loaded. Service ready.")
        except Exception as exc:  # noqa: BLE001 - surface any load failure via /health
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"[CHEM][api] Model load FAILED: {self.load_error}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full CHEM pipeline on any subset of modalities.

        Parameters
        ----------
        text : str, optional
            Raw text content.
        image_bytes : bytes, optional
            Raw image file bytes (decoded to a PIL image).
        audio_path : str, optional
            Path to a temporary audio file on disk.
        video_path : str, optional
            Path to a temporary video file on disk.

        Returns
        -------
        dict
            The full API response payload (see :func:`analyze_endpoint`).
        """
        if not self.loaded:
            raise RuntimeError(
                f"CHEM models are not loaded: {self.load_error or 'unknown error'}"
            )

        elements: List[Dict[str, Any]] = []
        modality_scores: Dict[str, float] = {}

        # --- Text ---
        if text:
            print("[CHEM][api] Encoding text...")
            res = self.text_encoder.encode(text)
            elements.extend(_reindex(res.get("elements", []), "text", len(elements)))
            modality_scores["text"] = float(res.get("harm_score", 0.0))

        # --- Image ---
        if image_bytes is not None:
            print("[CHEM][api] Encoding image...")
            from PIL import Image

            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            res = self.image_encoder.encode(pil)
            elements.extend(_reindex(res.get("elements", []), "image", len(elements)))
            modality_scores["image"] = float(res.get("harm_score", 0.0))

        # --- Audio ---
        if audio_path is not None:
            print("[CHEM][api] Encoding audio...")
            res = self.audio_encoder.encode(audio_path)
            elements.extend(_reindex(res.get("elements", []), "audio", len(elements)))
            modality_scores["audio"] = float(res.get("harm_score", 0.0))

        # --- Video ---
        if video_path is not None:
            print("[CHEM][api] Encoding video...")
            res = self.video_encoder.encode(video_path)
            elements.extend(_reindex(res.get("elements", []), "video", len(elements)))
            modality_scores["video"] = float(res.get("harm_score", 0.0))

        if not elements:
            return {
                "harm_score": 0.0,
                "verdict": "SAFE",
                "policy_clause": None,
                "triggered_category": None,
                "triggered_elements": [],
                "nodes": [],
                "edges": [],
                "explanation": "No content was provided to analyse.",
                "cross_modal_flag": False,
                "shap_scores": [],
                "modality_scores": {},
            }

        # --- Graph construction ---
        print("[CHEM][api] Building Content Composition Graph...")
        graph = self.graph_builder.build(elements)

        # --- GAT + pattern-library scoring ---
        print("[CHEM][api] Scoring graph...")
        score = self.harm_scorer.score(graph)

        # --- Exculpatory context dampening ---
        # If the text signals a benign real-world context (acting, audition, etc.)
        # dampen image and video scores before fusion so they don't override context.
        if _is_exculpatory(text):
            factor = get_coefficient("exculpatory_dampening_factor")
            for mod in ("image", "video"):
                if mod in modality_scores:
                    modality_scores[mod] = modality_scores[mod] * factor

        # --- Dynamic cross-modal fusion ---
        fused = self._fuse(modality_scores) if self._fuse else {
            "fused_score": score.get("harm_score", 0.0),
            "cross_modal_flag": False,
        }
        # Weighted blend: GAT graph score (70%) + modality fusion (30%).
        # Using max() was too aggressive — a single noisy encoder could flip the
        # verdict. A blend requires both signals to agree before scoring high.
        gat_score   = float(score.get("harm_score", 0.0))
        fused_score = float(fused.get("fused_score", 0.0))
        final_score = 0.7 * gat_score + 0.3 * fused_score
        threshold = get_coefficient("final_harm_threshold")
        verdict = "HARMFUL" if final_score >= threshold else "SAFE"

        matched_pattern = score.get("matched_pattern")
        policy_clause = score.get("policy_clause")
        triggered_category = None
        if isinstance(policy_clause, dict):
            triggered_category = policy_clause.get("clause_id")

        # --- Explainability (best-effort; never breaks the response) ---
        shap_scores: List[Any] = []
        explanation = ""
        try:
            shap_result = self.shap_explainer.explain(graph) if self.shap_explainer else {}
            gnn_result = self.gnn_explainer.explain(graph) if self.gnn_explainer else {}
            shap_scores = shap_result.get("node_scores", shap_result.get("attributions", []))
            report = self.report_generator.generate(
                graph_dict=graph,
                shap_result=shap_result,
                gnn_result=gnn_result,
                matched_pattern=matched_pattern,
                policy_clause=policy_clause,
            )
            explanation = report.get("explanation", "")
        except Exception as exc:  # noqa: BLE001
            print(f"[CHEM][api] Explainability degraded: {exc}")
            explanation = _fallback_explanation(policy_clause, score.get("triggered_elements", []))

        return {
            "harm_score": round(final_score, 4),
            "verdict": verdict,
            "policy_clause": policy_clause,
            "triggered_category": triggered_category,
            "triggered_elements": score.get("triggered_elements", []),
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
            "explanation": explanation,
            "cross_modal_flag": bool(fused.get("cross_modal_flag", False)),
            "shap_scores": shap_scores,
            "modality_scores": {k: round(v, 4) for k, v in modality_scores.items()},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reindex(elements: List[Dict[str, Any]], modality: str, offset: int) -> List[Dict[str, Any]]:
    """Ensure element ids are globally unique across modalities.

    Parameters
    ----------
    elements : list of dict
        Elements emitted by a single encoder.
    modality : str
        The modality name, used as an id prefix.
    offset : int
        Current number of elements already collected (keeps ids monotonic).

    Returns
    -------
    list of dict
        The same elements with rewritten globally-unique ``id`` fields; any
        ``target_id`` references are left intact for the graph builder to
        resolve within the encoder's own namespace.
    """
    out = []
    for i, el in enumerate(elements):
        el = dict(el)
        el.setdefault("modality", modality)
        el["id"] = f"{modality}_{offset + i}"
        out.append(el)
    return out


def _fallback_explanation(policy_clause: Any, triggered: List[Any]) -> str:
    """Produce a minimal explanation when the explainability stack is degraded.

    Parameters
    ----------
    policy_clause : dict or None
        The matched policy clause metadata.
    triggered : list
        Triggered element dictionaries.

    Returns
    -------
    str
        A short human-readable explanation string.
    """
    clause = (policy_clause or {}).get("title", "an unmapped policy") if isinstance(policy_clause, dict) else "an unmapped policy"
    return (
        f"This content was flagged under {clause}. "
        f"{len(triggered)} element(s) contributed to the decision. "
        "Detailed attribution was unavailable for this request."
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CHEM — Compositional Harm Emergence Modeling",
    description="Policy-aware multimodal content moderation with explainable decisions.",
    version=load_policy_config().get("meta", {}).get("version", "1.0.0"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

pipeline = CHEMPipeline()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.on_event("startup")
def _startup() -> None:
    """FastAPI startup hook: load all models exactly once."""
    pipeline.load()


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness endpoint.

    Returns
    -------
    dict
        ``{"status": "ok"|"degraded", "models_loaded": bool, "error": str|None}``.
    """
    return {
        "status": "ok" if pipeline.loaded else "degraded",
        "models_loaded": pipeline.loaded,
        "error": pipeline.load_error,
    }


@app.post("/analyze")
async def analyze_endpoint(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    """Analyse multimodal content and return a full explainable verdict.

    Accepts multipart form-data with any subset of ``text``, ``image``,
    ``audio`` and ``video`` fields.

    Returns
    -------
    JSONResponse
        ``{nodes, edges, harm_score, verdict, policy_clause, triggered_category,
        triggered_elements, explanation, cross_modal_flag, shap_scores,
        modality_scores}``.
    """
    if not pipeline.loaded:
        return JSONResponse(
            status_code=503,
            content={"error": f"Models not loaded: {pipeline.load_error}"},
        )

    tmp_paths: List[str] = []
    try:
        image_bytes = await image.read() if image is not None else None

        audio_path = None
        if audio is not None:
            audio_path = _spool(await audio.read(), suffix=os.path.splitext(audio.filename or "a.wav")[1])
            tmp_paths.append(audio_path)

        video_path = None
        if video is not None:
            video_path = _spool(await video.read(), suffix=os.path.splitext(video.filename or "v.mp4")[1])
            tmp_paths.append(video_path)

        result = pipeline.analyze(
            text=text,
            image_bytes=image_bytes,
            audio_path=audio_path,
            video_path=video_path,
        )
        return JSONResponse(content=_sanitize(result))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Infinity floats so the payload is JSON-safe.

    Python's ``json`` module rejects ``float('nan')`` and ``float('inf')``,
    which can appear in SHAP attributions or metric computations when a graph
    has no edges or a single node.  This replaces them with ``None`` so the
    response always serialises cleanly.

    Parameters
    ----------
    obj : Any
        Arbitrary nested structure (dict, list, float, …).

    Returns
    -------
    Any
        The same structure with NaN/Inf floats replaced by ``None``.
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _spool(data: bytes, suffix: str = "") -> str:
    """Write uploaded bytes to a temporary file and return its path.

    Parameters
    ----------
    data : bytes
        Raw uploaded file content.
    suffix : str
        File suffix (e.g. ``.wav``, ``.mp4``) so downstream loaders pick the
        right decoder.

    Returns
    -------
    str
        Absolute path to the spooled temporary file (caller must delete it).
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

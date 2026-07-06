"""Route (experiment) configuration schema and loader.

Routes are YAML descriptions merged over a shared base via OmegaConf and
materialized into frozen dataclasses for the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, cast

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    input_dim: int
    keep: int
    transform: str


@dataclass(frozen=True)
class DataConfig:
    modalities: List[ModalitySpec]
    latent_dim: int
    cohort_size: int
    max_observations: int
    horizon_days: float
    event_horizon_days: float
    censor_fraction: float
    active_modalities: List[str]


@dataclass(frozen=True)
class GrowthConfig:
    alpha: float
    carrying_capacity: float
    kill_rate: float
    saturation: float
    basal_influx: float
    stimulation: float
    half_max: float
    immune_death: float
    rho_immune: float
    eta_immune: float


@dataclass(frozen=True)
class DynamicsConfig:
    physics_dims: int
    residual_hidden: int
    residual_layers: int
    bound_weight: float
    rtol: float
    atol: float
    solver: str
    volume_max: float
    immune_max: float


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int
    heads: int
    encoder_hidden: int
    reward_hidden: int
    decay_init: float
    growth: GrowthConfig
    dynamics: DynamicsConfig


@dataclass(frozen=True)
class RLConfig:
    actions: List[int]
    gamma: float
    lambda_detection: float
    lambda_cost: float
    lambda_missed: float
    risk_threshold: float
    monitor_floor_days: int
    override_days: int
    policy_hidden: Tuple[int, int]
    rollout_steps: int
    step_days: float


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    restart_period: int
    restart_min_lr: float


@dataclass(frozen=True)
class ScheduleConfig:
    world_epochs: int
    policy_epochs: int
    batch_size: int
    grad_accum: int
    world_size: int
    patience: int
    precision: str


@dataclass(frozen=True)
class LossWeights:
    physics: float
    rl: float
    kl: float


@dataclass(frozen=True)
class AblationFlags:
    physics: bool
    imagination: bool
    fusion: bool
    drl: bool
    immune_coupling: bool
    cross_modal_attention: bool


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    rl: RLConfig
    optim: OptimConfig
    schedule: ScheduleConfig
    loss_weights: LossWeights
    ablation: AblationFlags
    extras: Dict[str, float] = field(default_factory=dict)

    @property
    def effective_batch(self) -> int:
        return (
            self.schedule.batch_size * self.schedule.grad_accum * self.schedule.world_size
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build(blob: Mapping[str, Any]) -> ExperimentConfig:
    g = blob["model"]["growth"]
    d = blob["model"]["dynamics"]
    modalities = [ModalitySpec(**m) for m in blob["data"]["modalities"]]
    active = blob["data"].get("active_modalities") or [m.name for m in modalities]
    data = DataConfig(
        modalities=modalities,
        latent_dim=int(blob["data"]["latent_dim"]),
        cohort_size=int(blob["data"]["cohort_size"]),
        max_observations=int(blob["data"]["max_observations"]),
        horizon_days=float(blob["data"]["horizon_days"]),
        event_horizon_days=float(blob["data"]["event_horizon_days"]),
        censor_fraction=float(blob["data"]["censor_fraction"]),
        active_modalities=[str(m) for m in active],
    )
    model = ModelConfig(
        latent_dim=int(blob["model"]["latent_dim"]),
        heads=int(blob["model"]["heads"]),
        encoder_hidden=int(blob["model"]["encoder_hidden"]),
        reward_hidden=int(blob["model"]["reward_hidden"]),
        decay_init=float(blob["model"]["decay_init"]),
        growth=GrowthConfig(**{k: float(v) for k, v in g.items()}),
        dynamics=DynamicsConfig(
            physics_dims=int(d["physics_dims"]),
            residual_hidden=int(d["residual_hidden"]),
            residual_layers=int(d["residual_layers"]),
            bound_weight=float(d["bound_weight"]),
            rtol=float(d["rtol"]),
            atol=float(d["atol"]),
            solver=str(d["solver"]),
            volume_max=float(d["volume_max"]),
            immune_max=float(d["immune_max"]),
        ),
    )
    r = blob["rl"]
    rl = RLConfig(
        actions=[int(a) for a in r["actions"]],
        gamma=float(r["gamma"]),
        lambda_detection=float(r["lambda_detection"]),
        lambda_cost=float(r["lambda_cost"]),
        lambda_missed=float(r["lambda_missed"]),
        risk_threshold=float(r["risk_threshold"]),
        monitor_floor_days=int(r["monitor_floor_days"]),
        override_days=int(r["override_days"]),
        policy_hidden=(int(r["policy_hidden"][0]), int(r["policy_hidden"][1])),
        rollout_steps=int(r["rollout_steps"]),
        step_days=float(r["step_days"]),
    )
    o = blob["optim"]
    optim = OptimConfig(
        lr=float(o["lr"]),
        weight_decay=float(o["weight_decay"]),
        beta1=float(o["beta1"]),
        beta2=float(o["beta2"]),
        grad_clip=float(o["grad_clip"]),
        restart_period=int(o["restart_period"]),
        restart_min_lr=float(o["restart_min_lr"]),
    )
    s = blob["schedule"]
    schedule = ScheduleConfig(
        world_epochs=int(s["world_epochs"]),
        policy_epochs=int(s["policy_epochs"]),
        batch_size=int(s["batch_size"]),
        grad_accum=int(s["grad_accum"]),
        world_size=int(s["world_size"]),
        patience=int(s["patience"]),
        precision=str(s["precision"]),
    )
    lw = blob["loss_weights"]
    weights = LossWeights(
        physics=float(lw["physics"]), rl=float(lw["rl"]), kl=float(lw["kl"])
    )
    ab = blob["ablation"]
    ablation = AblationFlags(
        physics=bool(ab["physics"]),
        imagination=bool(ab["imagination"]),
        fusion=bool(ab["fusion"]),
        drl=bool(ab["drl"]),
        immune_coupling=bool(ab["immune_coupling"]),
        cross_modal_attention=bool(ab["cross_modal_attention"]),
    )
    extras = {str(k): float(v) for k, v in dict(blob.get("extras", {})).items()}
    return ExperimentConfig(
        name=str(blob["name"]),
        seed=int(blob["seed"]),
        data=data,
        model=model,
        rl=rl,
        optim=optim,
        schedule=schedule,
        loss_weights=weights,
        ablation=ablation,
        extras=extras,
    )


def load_route(route: str) -> ExperimentConfig:
    route_path = Path(route)
    if not route_path.is_absolute():
        candidate = _repo_root() / route
        route_path = candidate if candidate.exists() else route_path
    base = OmegaConf.load(_repo_root() / "beta" / "_base.yaml")
    merged = OmegaConf.merge(base, OmegaConf.load(route_path))
    blob = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(blob, dict):
        raise ValueError(f"route {route} did not resolve to a mapping")
    return _build(cast(Mapping[str, Any], blob))

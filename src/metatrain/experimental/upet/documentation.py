from typing import Literal, Optional
from typing_extensions import TypedDict
from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights
from metatrain.pet.modules.finetuning import FinetuneHypers, NoFinetuneHypers

class ModelHypers(TypedDict):
    """Hyperparameters for the UPET model."""
    cutoff: float = 5.0
    d_pet: int = 128
    d_node: int = 256
    d_head: int = 128
    d_feedforward: int = 256
    num_heads: int = 8
    num_attention_layers: int = 2
    num_gnn_layers: int = 2
    normalization: Literal["RMSNorm", "LayerNorm"] = "RMSNorm"
    activation: Literal["SiLU", "SwiGLU"] = "SwiGLU"
    transformer_type: Literal["PreLN", "PostLN"] = "PreLN"
    attention_temperature: float = 1.0
    featurizer_type: Literal["residual", "feedforward"] = "feedforward"
    cutoff_width: float = 0.5
    
    # Compilation flag
    compile: bool = True
    """Whether to use torch.compile on the core model."""

class TrainerHypers(TypedDict):
    """Hyperparameters for training UPET models."""
    distributed: bool = False
    distributed_port: int = 39591
    batch_size: int = 16
    num_epochs: int = 100
    warmup_fraction: float = 0.05
    learning_rate: float = 1e-4
    weight_decay: Optional[float] = 1e-5
    
    log_interval: int = 1
    checkpoint_interval: int = 10
    
    atomic_baseline: FixedCompositionWeights = {}
    scale_targets: bool = True
    fixed_scaling_weights: FixedScalerWeights = {}
    per_structure_targets: list[str] = []
    
    num_workers: Optional[int] = None
    log_mae: bool = True
    log_separate_blocks: bool = False
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    grad_clip_norm: float = 1.0
    loss: str | dict[str, LossSpecification | str] = "mse"
    batch_atom_bounds: list[Optional[int]] = [None, None]
    
    finetune: NoFinetuneHypers | FinetuneHypers = {
        "read_from": None, 
        "method": "full", 
        "config": {}, 
        "inherit_heads": {}
    }

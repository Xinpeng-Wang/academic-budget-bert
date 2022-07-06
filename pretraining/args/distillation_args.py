from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DistillationArguments:
    """
    Distillation configuration arguments
    """

    _argument_group_name = "Distillation Arguments"

    distllation: Optional[bool] = field(
        default=False, metadata={"help": "whether to do distillation"}
    )

    fearture_learn: Optional[str] = field(
        default=None, metadata={"help": "which feature distillation method to use"}
    )

    
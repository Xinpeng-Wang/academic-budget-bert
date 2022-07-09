from csv import field_size_limit
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DistillationArguments:
    """
    Distillation configuration arguments
    """

    _argument_group_name = "Distillation Arguments"

    distillation: Optional[bool] = field(
        default=False, metadata={"help": "whether to do distillation"}
    )

    fearture_learn: Optional[str] = field(
        default=None, metadata={"help": "which feature distillation method to use"}
    )

    teacher_path: Optional[str] = field(
        default=None, metadata={"help": "the path of the teacher model"}
    )

    layer_selection: Optional[lambda s: [int(item) for item in s.split(',')]] = field(
        default=None,
        metadata={"help": "the layer of teacher model to learn from" }
    )
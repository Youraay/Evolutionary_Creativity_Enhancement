"""Paths and variants of the experiment archive_size."""
from experiments.common.settings import db_file, result_dir
from experiments.common.variants import Variant, VariantExperiment

EXPERIMENT = VariantExperiment(
    name="archive_size",
    variants=[
        Variant("No_Archiv"),
        Variant("Archiv_Low", overrides={"fitness": "mmd_archive:200:0.0"}),
        Variant("Archiv_Mid", overrides={"fitness": "mmd_archive:200:0.08"}),
        Variant("Archiv_High", overrides={"fitness": "mmd_archive:200:0.1"}),
        Variant("Full_Archiv", overrides={"fitness": "mmd_archive:6000:0.0"}),
    ],
    db_file=db_file("archive_size"),
    result_dir=result_dir("archive_size"),
)

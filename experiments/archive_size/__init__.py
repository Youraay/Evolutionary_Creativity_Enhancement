"""Archive size: novelty fitness with an archive of earlier candidates ("a cat", seeds 1-3).

Supplementary Fig. 6.

Variants (configuration of every run in ``config.py`` and in the job database):

    No_Archiv             no archive (standard)
    Archiv_Low/Mid/High   archive of 200 candidates; a candidate enters if its novelty exceeds 0.0 / 0.08 / 0.1
    Full_Archiv           archive of 6000 candidates, i.e. every candidate of the run (100 x 30 = 3000)

Standard configuration: population 100, 30 generations, SDXL with 50 steps, tournament size 3, uniform
crossover (rate 0.9), Gaussian mutation of 10 % of the values with standard deviation 0.2 (probability 0.05),
elitism 1, novelty fitness in BLIP-2 space (see ``experiments.common.variants``).

CLIP scores of the images are in ``<result dir>/clip_scores``.

    python -m experiments.archive_size.create_jobs --db new_jobs.db       # open jobs for all variants
    ARCHIVE_SIZE_DB_FILE=new_jobs.db python -m experiments.archive_size.dispatch

``data/jobs/archive_size.db`` lists the analysed runs (``create_jobs --register``).
"""

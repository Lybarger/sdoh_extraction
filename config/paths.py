


import os

username = os.getlogin()

"""
Configurable paths
"""


sdoh_corpus_challenge = f'/home/{username}/data/social_determinants_challenge/'

analyses = f'/home/{username}/sdoh_challenge/analyses/'



"""
Fixed paths
"""

brat_import = os.path.join(analyses, "step010_brat_import")
data_checks = os.path.join(analyses, "step012_data_checks")

tokenization_checks = os.path.join(analyses, "step013_tokenization_checks")
extraction  = os.path.join(analyses, "step110_extraction")
extraction_multi  = os.path.join(analyses, "step111_multi_spert_train")
multi_spert_eval = os.path.join(analyses,  "step112_multi_spert_infer")

multitask_train = os.path.join(analyses, "step121_multitask_train")
multitask_infer = os.path.join(analyses, "step122_multitask_infer")

summary  = os.path.join(analyses, "step120_summary")
compare_scoring_criteria = os.path.join(analyses, "step122_compare_scoring_criteria")

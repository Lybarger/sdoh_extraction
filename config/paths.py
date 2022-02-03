


import os



"""
Input paths
"""

sdoh_corpus_review = '/home/lybarger/data/social_determinants_review/'
sdoh_corpus_challenge = '/home/lybarger/data/social_determinants_challenge/'

"""
Output paths
"""
username = os.getlogin()

analyses = f'/home/{username}/sdoh_challenge/analyses/'


brat_import = os.path.join(analyses, "step010_brat_import")
extraction  = os.path.join(analyses, "step110_extraction")
summary  = os.path.join(analyses, "step120_summary")
compare_scoring_criteria = os.path.join(analyses, "step122_compare_scoring_criteria")

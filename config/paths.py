


import os



"""
Input paths
"""

brat_annotation = '/home/lybarger/clinical_extractors/analyses_deid/step020_brat_deid_diff/sdoh/social_determinants_review/'


"""
Output paths
"""
username = os.getlogin()

analyses = f'/home/{username}/sdoh_challenge/analyses/'


brat_import = os.path.join(analyses, "step010_brat_import")
extraction  = os.path.join(analyses, "step110_extraction")

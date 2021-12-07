


import os



"""
Input paths
"""

brat_radiology_anatomy = '/home/data/bionlp/anatomy_classes/'
#brat_radiology_events = '/home/data/bionlp/UW_500/'
#brat_radiology_events = '/home/data/bionlp/UW_500_deid/'

brat_radiology_events = '/home/data/bionlp/UW_500_deid_CT/'

incidentaloma_6k = '/home/data/bionlp/incidentaloma/deid_6000_random_sampled_json/'

incidentalomas_round1 = "/home/data/bionlp/incidentalomas/round01/"
incidentalomas = "/home/data/bionlp/incidentalomas/"



"""
Output paths
"""
username = os.getlogin()

analyses = f'/home/{username}/incidentalomas/analyses/'


text_import = os.path.join(analyses, "step005_text_import")
text_sample = os.path.join(analyses, "step007_text_sample")
whitespace_adjust = os.path.join(analyses, "step009_whitespace_adjust")
brat_import = os.path.join(analyses, "step010_brat_import")
annotation_qc = os.path.join(analyses, "step011_annotation_qc")
annotator_agreement = os.path.join(analyses, "step012_annotator_agreement")
agreement_summary = os.path.join(analyses, "step013_agreement_summary")
annotation_merge = os.path.join(analyses, "step016_annotation_merge")
spert_datasets = os.path.join(analyses, "step020_spert_datasets")
anatomy_norm_modeling = os.path.join(analyses, "step101_anatomy_norm_modeling")
anatomy_extraction = os.path.join(analyses, "step102_anatomy_extraction")
pipeline_extraction = os.path.join(analyses, "step103_pipeline_extraction")

anatomy_norm_summary = os.path.join(analyses, "step121_anatomy_norm_summary")
anatomy_extraction_summary = os.path.join(analyses, "step122_anatomy_extraction_summary")
sampling = os.path.join(analyses, "step201_sampling")

anatomy_data_figures = os.path.join(analyses, "step124_anatomy_data_figures")
anatomy_norm_figures = os.path.join(analyses, "step125_anatomy_norm_figures")
result_figures = os.path.join(analyses, "step126_result_figures")

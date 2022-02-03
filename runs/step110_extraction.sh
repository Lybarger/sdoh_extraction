#!/bin/bash




python3 runs/step110_extraction.py with fast_run=True description='sdoh_challenge_5_d2' source="sdoh_challenge" epochs=5 prop_drop=0.2


python3 runs/step110_extraction.py with fast_run=False description='sdoh_challenge_25_d2' source="sdoh_challenge" epochs=25 prop_drop=0.2 device=0
python3 runs/step110_extraction.py with fast_run=False description='sdoh_review_25_d2' source="sdoh_review" epochs=25 prop_drop=0.2 device=1

# python3 runs/step110_extraction.py with fast_run=False description='e01_d2' epochs=1  prop_drop=0.2
# python3 runs/step110_extraction.py with fast_run=False description='e04_d2' epochs=4  prop_drop=0.2
# python3 runs/step110_extraction.py with fast_run=False description='e08_d2' epochs=8  prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e10_d2' epochs=10 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e15_d2' epochs=15 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e20_d2' epochs=20 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e25_d2' epochs=25 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e30_d2' epochs=30 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e35_d2' epochs=35 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e40_d2' epochs=40 prop_drop=0.2
python3 runs/step110_extraction.py with fast_run=False description='e45_d2' epochs=45 prop_drop=0.2

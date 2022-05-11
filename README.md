# Social Determinants of Health


## BRAT Configuration
The BRAT configuration files are available at [annotation.conf](resources/annotation.conf) and [visual.conf](resources/visual.conf).


## Scoring Routine
The evaluation criteria are defined in [sdoh_scoring.pdf](resources/sdoh_scoring.pdf).

## Implementation details

* *Tokenization* - Sentence boundary detection and word tokenization should be performed using the *default English spaCy tokenizer*.
* *Trigger matches* - A predicted trigger can match *at most* one gold trigger, and a gold trigger can match *at most* one predicted trigger. In other words, a predicted trigger cannot be considered equivalent to multiple gold triggers, and multiple predicted triggers cannot be considered equivalent to a single gold trigger.
* *Argument matches* - An argument may be connected to multiple triggers (part of multiple events). Arguments are assessed through trigger-argument pairs, rather than individual arguments. A predicted trigger-argument pair can match *at most* one gold trigger-argument pair, and a gold trigger-argument pair can match *at most* one predicted trigger-argument pair.



The annotated corpus in BRAT format can be import using the `step010_brat_import.py` script. 
```
python3 runs/step010_brat_import.py
```



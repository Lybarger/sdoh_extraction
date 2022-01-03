# Social Determinants of Health


## BRAT Configuration
The BRAT configuration files are available at [annotation.conf](resources/annotation.conf) and [visual.conf](resources/visual.conf).


## Scoring Routine
The evaluation criteria are defined in [sdoh_scoring.pdf](resources/sdoh_scoring.pdf).

## Implementation details

* *Tokenization* - Sentence boundary detection and word tokenization should be performed using the default English spaCy tokenizer
* *One-to-one matches* - A predicted trigger or argument can match at most one gold trigger or argument. A predicted trigger/argument cannot be considered equivalent to multiple gold trigger/arguments. Multiple predicted triggers/arguments cannot be considered equivalent to a single gold trigger/argument.

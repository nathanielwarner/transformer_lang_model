# LeClair "FunCom" Java dataset
A filtered dataset consisting of 2.1 million Java function/documentation
pairs. The project that each function comes from is kept track of
in order to prevent dataset leakage. (i.e. the functions in the test
set should not come from the same project as any in the training set)

### Instructions
1. Download the filtered dataset (not tokenized or raw) from 
[here](http://leclair.tech/data/funcom/).
2. Run `process_funcom.py` to split into train/val/test.
3. Run `create_spm.py` to create the SentencePiece models

### Citation
&mdash; <cite>LeClair, A., McMillan, C., "Recommendations for Datasets for Source Code Summarization", 
in Proc. of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics 
(NAACL'19), Short Research Paper Track, Minneapolis, USA, June 2-7, 2019.</cite>

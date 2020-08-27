import sentencepiece as spm


spm.SentencePieceTrainer.Train(input="train_codes.txt", model_prefix="code_spm", model_type="unigram", vocab_size=8192,
                               character_coverage=1.0)
spm.SentencePieceTrainer.Train(input="train_nl.txt", model_prefix="nl_spm", model_type="unigram", vocab_size=8192,
                               character_coverage=1.0)

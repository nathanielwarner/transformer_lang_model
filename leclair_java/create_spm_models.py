import sentencepiece as spm


spm.SentencePieceTrainer.Train('--input=train_codes.txt --model_prefix=code_spm --model_type=bpe --vocab_size=8192')
spm.SentencePieceTrainer.Train('--input=train_nl.txt --model_prefix=nl_spm --model_type=bpe --vocab_size=8192')

# -*- coding: utf-8 -*-



from fastai.text import *
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import sentencepiece as spm
import re
import pdb

import fastai, torch
fastai.__version__ , torch.__version__

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

random_seed(42, True)

path = Path('./')

df_train = pd.read_csv(path/'pa-train.csv', header=None)
df_train.head()

df_valid = pd.read_csv(path/'pa-valid.csv', header=None)
df_valid.head()

df_test = pd.read_csv(path/'pa-test.csv', header=None)
df_test.head()

df_train.shape, df_valid.shape, df_test.shape

df_train[df_train[0].isnull()].shape, df_valid[df_valid[0].isnull()].shape, df_test[df_test[0].isnull()].shape

label_cols = [0]

class PanjabiTokenizer(BaseTokenizer):
    def __init__(self, lang:str):
        self.lang = lang
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str('panjabi_lm.model'))
        
    def tokenizer(self, t:str) -> List[str]:
        return self.sp.EncodeAsPieces(t)

sp = spm.SentencePieceProcessor()
sp.Load('panjabi_lm.model')
itos = [sp.IdToPiece(int(i)) for i in range(30000)]

# 30,000 is the vocab size that we chose in sentencepiece
panjabi_vocab = Vocab(itos)

panjabi_tok = PanjabiTokenizer('pb')

tokenizer = Tokenizer(tok_func=PanjabiTokenizer, lang='pb')

tokenizer.special_cases

data_lm = TextLMDataBunch.from_df(path=path, train_df=df_train, valid_df=df_valid, test_df=df_test, tokenizer=tokenizer, vocab=panjabi_vocab, label_cols=label_cols)

data_lm.show_batch()

learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.3, pretrained=False)

# Loading the pretrained language model on panjabi wikipedia
learn.load(path/'modell', with_opt=True)


learn.freeze()

learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()

learn.fit_one_cycle(3, 1e-3)

learn.predict('ਜੋ ਉਹਨਾਂ ਦੇ ਰੱਬਾਂ ਨੂੰ ਪ੍ਰਸਤੁਤ ਕਰਦੇ',n_words=10)

learn.save_encoder('fine_tuned_enc')

data_clas = TextClasDataBunch.from_df(path=path, train_df=df_train, valid_df=df_valid, test_df=df_test, tokenizer=tokenizer, vocab=panjabi_vocab, label_cols=label_cols, bs=64)

data_clas.show_batch()

learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')

learn.freeze()

learn.loss_func.func

mcc = MatthewsCorreff()

learn.metrics = [accuracy, mcc]

learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(5, 1e-3, callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='accuracy', name='final')])

learn.load('final')

from sklearn.metrics import accuracy_score, matthews_corrcoef
df_dict = {'query': list(df_test[1]), 'actual_label': list(df_test[0]), 'predicted_label': ['']*df_test.shape[0]}
all_nodes = list(set(df_train[0]))
for node in all_nodes:
    df_dict[node] = ['']*df_test.shape[0]
    
i2c = {}
for key, value in learn.data.c2i.items():
    i2c[value] = key
    
df_result = pd.DataFrame(df_dict)
preds = learn.get_preds(ds_type=DatasetType.Test, ordered=True)
for index, row in df_result.iterrows():
    for node in all_nodes:
        row[node] = preds[0][index][learn.data.c2i[node]].item()
    row['predicted_label'] = i2c[np.argmax(preds[0][index]).data.item()]
df_result.head()

accuracy_score(df_result['actual_label'], df_result['predicted_label'])

matthews_corrcoef(df_result['actual_label'], df_result['predicted_label'])

df_result.to_csv('indicnlp_news_articles_pa_result.csv', index=False)


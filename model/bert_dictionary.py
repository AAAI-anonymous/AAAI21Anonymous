"""
TODO: load_from_file -> load

加入到 https://github.com/pytorch/fairseq/blob/master/fairseq/data/legacy/masked_lm_dictionary.py
"""

from fairseq.data import Dictionary
from fairseq.data import data_utils
import torch
from fairseq import utils

class BertDictionary(Dictionary):

    @classmethod
    def load_from_file(cls, filename):
        d = cls()
        d.symbols = []
        d.count = []
        d.indices = {}

        nspecial = 0
        with open(filename, 'r', encoding='utf-8', errors='ignore') as input_file:
            for line in input_file:  #
                k = line.rstrip('\n').rsplit(' ', 1)[0]
                if (k[0] == '[' and k[-1] == ']') or k in ['<S>', '<T>']:
                    nspecial += 1
                d.add_symbol(k)

        d.unk_word = '[UNK]'
        d.pad_word = '[PAD]'
        d.eos_word = '[SEP]'
        d.bos_word = '[CLS]'

        d.bos_index = d.add_symbol('[CLS]')
        d.pad_index = d.add_symbol('[PAD]')
        d.eos_index = d.add_symbol('[SEP]')
        d.unk_index = d.add_symbol('[UNK]')
        d.mask_index = d.add_symbol('[MASK]')

        d.nspecial = nspecial  # en 999, zh 106
        return d

    def save(self, f):
        """Stores dictionary into a text file
        """
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols, ex_vals + self.count))
    
    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        #extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = " ".join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )

        return data_utils.post_process(sent, bpe_symbol)

    # def cls(self):
    #     """Helper to get index of cls symbol"""
    #     return self.cls_index
    #
    # def sep(self):
    #     """Helper to get index of sep symbol"""
    #     return self.sep_index

import os
import logging
import numpy as np
import codecs
from typing import Dict, List, Iterable, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

start_token = "[CLS]"
sep_token = "[SEP]"


@DatasetReader.register("bert")
class BERTReader(DatasetReader):
    """
    This DatasetReader is designed to read in the pre-processed BERT data for training
    BERT models. It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The WordPiece tokens in the sentence.
    segment_ids : ``SequenceLabelField``
        The labels of each of the tokens (0 - tokens from the first sentence,
        1 - tokens from the second sentence).
    # masked_lm_positions : ``SequenceLabelField``
    #     For each token, whether it is masked or not.
    lm_label_ids : ``SequenceLabelField``
        For each masked position, what is the correct label.
    next_sentence_label : ``LabelField``
        Next sentence label: is the second sentence the next sentence following the
        first one, or is it a randomly selected sentence.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    # token_indexers : ``Dict[str, TokenIndexer]``, optional
    #     We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    #     Default is ``{"tokens": PretrainedBertIndexer()}``.
    bert_vocab: ``str``, (default = None)
        A string denoting the bert-vocab type. Options are:
        bert-base-cased
        bert-base-multilingual-cased
        bert-base-uncased
        bert-large-uncased
        bert-base-chinese
        bert-base-multilingual-uncased
        bert-large-cased
    max_predictions_per_seq: ``int``
        Maximum number of masked LM predictions per sequence.
    masked_lm_prob: ``float``
        Masked LM probability.


    Returns
    -------
    A ``Dataset`` of ``Instances`` for BERT pre-training.

    """
    def __init__(self,
                 # token_indexers: Dict[str, TokenIndexer] = None,
                 bert_vocab: str = "bert-base-uncased",
                 max_predictions_per_seq: int = 20,
                 masked_lm_prob: float = 0.15,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = {"tokens": PretrainedBertIndexer(bert_vocab)}
        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = masked_lm_prob

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading BERT instances from dataset files at: %s", file_path)

        for file in self._bert_subset(file_path):
            with codecs.open(file, 'r', encoding='utf8') as open_file:
                for sentence in open_file:
                    # Each sentence contains the following (tab-separated) fields:
                    # - target sequence length
                    # - label
                    # - sentence1
                    # - sentence2
                    fields = sentence.split("\t")

                    assert len(fields) == 4, "Number of files in {} is {} != 4".format(sentence, len(fields))

                    target_sequence_length = int(fields[0]) - 3
                    label = int(fields[1])
                    sentence1 = fields[2]
                    sentence2 = fields[3]

                    yield self.text_to_instance(target_sequence_length, label, sentence1, sentence2)

    @staticmethod
    def _bert_subset(file_path: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                yield os.path.join(root, data_file)

    def text_to_instance(self,  # type: ignore
                         target_sequence_length: int,
                         label: int,
                         sentence1: str,
                         sentence2: str) -> Instance:
        """
        We take two sentences here and first tokenize them.
        Then we trim their joint length to the desired sequence length.
        Finally, we mask them based on the BERT procedure.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        fields['next_sentence_label'] = LabelField(label, skip_indexing=True)

        # @todo: do we want to support anything other than word pieces? if so, how?
        sentence1_field = self._token_indexers["tokens"].wordpiece_tokenizer(sentence1)
        sentence2_field = self._token_indexers["tokens"].wordpiece_tokenizer(sentence2)


        tokens, segment_ids, lm_label_ids = \
            self.create_token_field(sentence1_field, sentence2_field, target_sequence_length)

        tokens = [Token(t) for t in tokens]

        tokens_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = tokens_field
        fields['segment_ids'] = SequenceLabelField(segment_ids, tokens_field)
        fields['lm_label_ids'] = SequenceLabelField(lm_label_ids, tokens_field)

        return Instance(fields)


    def create_token_field(self,
                           tokens1: List[Token],
                           tokens2: List[Token],
                           max_sequence_length: int) -> Tuple[List[str], List[int], List[int]]:
        """
        Code largely based on Google's BERT implementation
        :param tokens1: first sentence tokens.
        :param tokens2: second sentence tokens.
        :param target_sequence_length: Target length for this sentence.
        :return: sample tokens, token types (first or second sentences),
                 positions of masked tokens (for LM objective function),
                 labels for masked tokens
        """
        tokens1, tokens2 = self.truncate_seq_pair(tokens1, tokens2, max_sequence_length)

        tokens = [start_token] + tokens1 + [sep_token] + tokens2 + [sep_token]
        segment_ids = (len(tokens1)+ 2) * [0] + (len(tokens2)+ 1) * [1]

        tokens, lm_label_ids = self.create_masked_lm_predictions(tokens)

        return tokens, segment_ids, lm_label_ids



    def create_masked_lm_predictions(self, tokens: List[Token]) -> Tuple[List[str], List[int]]:
      """Creates the predictions for the masked LM objective."""

      cand_indexes = []
      for (i, token) in enumerate(tokens):
        if token == start_token or token == sep_token:
          continue
        cand_indexes.append(i)

      np.random.shuffle(cand_indexes)

      output_tokens = list(tokens)

      num_to_predict = min(self.max_predictions_per_seq,
                           max(1, int(round(len(tokens) * self.masked_lm_prob))))

      masked_lms = []

      vocab = self._token_indexers["tokens"].ids_to_tokens

      covered_indexes = set()
      for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
          break
        if index in covered_indexes:
          continue
        covered_indexes.add(index)

        # 80% of the time, replace with [MASK]
        if np.random.random() < 0.8:
          masked_token = "[MASK]"
        else:
          # 10% of the time, keep original
          if np.random.random() < 0.5:
            masked_token = tokens[index]
          # 10% of the time, replace with random word
          else:
            masked_token = vocab[np.random.randint(0, len(vocab) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append([index, self._token_indexers["tokens"].vocab[tokens[index]]])

      masked_lms = sorted(masked_lms, key=lambda x: x[0])

      lm_label_ids = len(tokens) * [-1]
      for p in masked_lms:
        lm_label_ids[p[0]] = p[1]

      return output_tokens, lm_label_ids



    def truncate_seq_pair(self,
                          tokens1: List[Token],
                          tokens2: List[Token],
                          max_sequence_length: int) -> Tuple[List[Token], List[Token]]:
        """
        Truncates a pair of sequences to a target sequence length.
        Largely based on Google BERT code.

        :param tokens1: Tokens for first senetnece
        :param tokens2: Tokens for sentence senetnece
        :param target_sequence_length: maximum joint length
        :return: tokens1 and tokens2, after trimmed jointly to the correct length.
        """

        total_length = len(tokens1) + len(tokens2)

        if total_length <= max_sequence_length:
            return tokens1, tokens2

        start = int((len(tokens1) + len(tokens2)  - max_sequence_length) / 2)

        if len(tokens1) > len(tokens2):
            assert len(tokens1) >= 1

            end = (start + max_sequence_length - len(tokens2))

            return tokens1[start:end], tokens2
        else:
            assert len(tokens2) >= 1

            end = (start + max_sequence_length - len(tokens1))

            return tokens1, tokens2[start:end]


# pylint: disable=no-self-use,invalid-name
import pytest
import numpy

from allennlp.data.dataset_readers.bert_reader import BERTReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestBertReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        bert_reader = BERTReader(lazy=lazy, max_predictions_per_seq=0)
        instances = bert_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'bert_pretraining' )
        instances = ensure_list(instances)

        assert(len(instances) == 6)

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        # Test trimming second sentence
        assert tokens == ["[CLS]", "chapter", "eleven", "[SEP]", "two", "three", "re", "##em", "##pha", "[SEP]"]

        assert fields["is_next"].label == 0

        token_types = fields['token_types'].labels
        masked_lm_labels = fields['masked_lm_labels'].labels

        # Test token types
        assert token_types == 4 *[0] + 6 * [1]
        assert masked_lm_labels == 10 *[-1]


        fields = instances[3].fields
        tokens = [t.text for t in fields['tokens'].tokens]

        print(tokens)
        # Test trimming first sentence
        assert tokens == ["[CLS]", "one", "two", "[SEP]", "re", "##em", "##pha", "##si", "##ze", "[SEP]"]

        assert fields["is_next"].label == 0

        token_types = fields['token_types'].labels
        masked_lm_labels = fields['masked_lm_labels'].labels

        # Test token types
        assert token_types == 4 *[0] + 6 * [1]
        assert masked_lm_labels == 10 *[-1]


        fields = instances[4].fields
        tokens = [t.text for t in fields['tokens'].tokens]

        print(tokens)
        # Test trimming first sentence
        assert tokens == ["[CLS]", "chapter", "eleven", "head", "[SEP]", "one", "two", "three", "re", "##em", "##pha", "##si", "##ze", "joint", "here", "blue", "sky", "[SEP]"]

        assert fields["is_next"].label == 1

        token_types = fields['token_types'].labels
        masked_lm_labels = fields['masked_lm_labels'].labels

        # Test token types
        assert token_types == 5 *[0] + 13 * [1]
        assert masked_lm_labels == 18 *[-1]


    def test_masking(self):
        numpy.random.seed(111)
        bert_reader = BERTReader(max_predictions_per_seq=1)
        instances = bert_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'bert_pretraining' )
        instances = ensure_list(instances)

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        print(tokens)
        # Test trimming second sentence
        assert tokens == ["[CLS]", "[MASK]", "eleven", "[SEP]", "two", "three", "re", "##em", "##pha", "[SEP]"]

        assert fields["is_next"].label == 0

        token_types = fields['token_types'].labels
        masked_lm_labels = fields['masked_lm_labels'].labels

        # Test token types
        assert token_types == 4 *[0] + 6 * [1]
        assert masked_lm_labels == [-1] + [3127] + 8 * [-1]


        bert_reader = BERTReader(max_predictions_per_seq=2)
        instances = bert_reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'bert_pretraining' )
        instances = ensure_list(instances)

        fields = instances[2].fields
        tokens = [t.text for t in fields['tokens'].tokens]
        print(tokens)
        # Test trimming second sentence
        assert tokens == ["[CLS]", "[MASK]", "eleven", "[SEP]", "two", "three", "re", "##em", "[MASK]", "[SEP]"]

        assert fields["is_next"].label == 0

        token_types = fields['token_types'].labels
        masked_lm_labels = fields['masked_lm_labels'].labels

        # Test token types
        assert token_types == 4 *[0] + 6 * [1]
        assert masked_lm_labels == [-1] + [3127] + 6 * [-1] + [21890] + [-1]

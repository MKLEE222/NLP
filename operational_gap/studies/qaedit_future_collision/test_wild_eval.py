import unittest
from wild_eval import normalize_answer, exact_match_score, transcript_key, locality_preserved

class WildAlignedChecks(unittest.TestCase):
    def test_articles_punctuation(self):
        self.assertEqual(normalize_answer("The Paris."), "paris")
        self.assertEqual(normalize_answer("An Oxford!"), "oxford")
    def test_wild_normalized_exact_match(self):
        self.assertEqual(exact_match_score("The Eiffel Tower.", "Eiffel Tower"), 1)
        self.assertEqual(exact_match_score("Paris France", "Paris"), 0)
    def test_preflight_contract_is_not_answer_normalizer(self):
        self.assertNotEqual(transcript_key("The Eiffel Tower."),
                            transcript_key("Eiffel Tower"))
    def test_locality_checks_change_not_ground_truth(self):
        self.assertEqual(locality_preserved("cat", "cat"), 1)
        self.assertEqual(locality_preserved("cat", "dog"), 0)

if __name__ == "__main__":
    unittest.main()

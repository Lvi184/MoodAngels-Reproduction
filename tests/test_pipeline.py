from moodangels.pipeline import MoodAngelsPipeline
from moodangels.data import load_cases

def test_predict_first_case():
    cases = load_cases('data/syn_train.json')
    pipe = MoodAngelsPipeline('data/syn_train.json')
    res = pipe.diagnose_dict(cases[0].raw, agent='multi')
    assert res.label in (0,1)
    assert res.single_agent_results

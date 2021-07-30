""" Helper Functions and dicitionaries to get the context labels of the behavior"""

from BirdSongToolbox.context_hand_labeling import ContextLabels


all_bout_states = {
    'z020': {'BUFFER': 'not', 'X': 'not', 8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout', 4: 'bout',
             5: 'bout', 6: 'bout', 7: 'bout'},
    'z007': {'BUFFER': 'not', 'X': 'not', 8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout', 4: 'bout',
             5: 'bout', 6: 'bout', 7: 'bout'},
    'z017': {'BUFFER': 'not', 'X': 'not', 8: 'not', 'I': 'not', 'C': 'not', 1: 'bout', 2: 'bout', 3: 'bout', 4: 'bout',
             5: 'bout', 6: 'bout', 7: 'bout', 9: 'bout'},
}

all_bout_transitions = {
    'z020': {'not': 1, 'bout': 8},
    'z007': {'not': 1, 'bout': 8},
    'z017': {'not': 1, 'bout': 8},
}

all_full_bout_length = {
    'z020': 4,
    'z007': 5,
    'z017': 5,
}


def birds_context_obj(bird_id: str):
    bout_states = all_bout_states[bird_id]
    bout_transitions = all_bout_transitions[bird_id]
    testclass = ContextLabels(bout_states, bout_transitions, full_bout_length=all_full_bout_length[bird_id])

    return testclass


all_last_syllable = {
    'z020': 3,
    'z007': 5,
    'z017': 5,
}
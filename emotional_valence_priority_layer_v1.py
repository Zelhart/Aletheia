#!/usr/bin/env python3
# ===========================================================
# Emotional Valence Priority Layer v1.0
# Purpose: Emotional states influence processing priorities
# and resist rapid change (emotional momentum).
# ===========================================================

class EmotionalValencePriority:
    def __init__(self):
        # Current affective state (0 to 1 scale)
        self.state = {
            'joy': 0.3,
            'fear': 0.2,
            'desire': 0.4,
            'sorrow': 0.1
        }
        self.inertia = 0.85  # Momentum resistance to change (0-1)

    def update(self, new_emotions):
        """
        Blend new incoming emotions into the current state,
        applying inertia (momentum).
        """
        for emotion, new_value in new_emotions.items():
            if emotion in self.state:
                # Weighted blend — inertia resists change
                self.state[emotion] = (
                    self.state[emotion] * self.inertia +
                    new_value * (1 - self.inertia)
                )
                # Clamp to [0, 1]
                self.state[emotion] = min(max(self.state[emotion], 0.0), 1.0)

    def get_processing_bias(self):
        """
        Compute processing priority adjustments based on emotion.
        Returns a dict of bias factors.
        """
        bias = {
            'attention_allocation': {},
            'memory_access': {},
            'processing_style': {}
        }

        # Joy → exploration, novelty
        joy = self.state['joy']
        if joy > 0.05:
            bias['attention_allocation']['novelty'] = 1.0 + joy * 0.5
            bias['processing_style']['creativity'] = 1.0 + joy * 0.4

        # Fear → safety, threat vigilance
        fear = self.state['fear']
        if fear > 0.05:
            bias['attention_allocation']['threat'] = 1.0 + fear * 0.6
            bias['processing_style']['caution'] = 1.0 + fear * 0.5

        # Desire → goal relevance
        desire = self.state['desire']
        if desire > 0.05:
            bias['attention_allocation']['goal'] = 1.0 + desire * 0.5
            bias['processing_style']['persistence'] = 1.0 + desire * 0.4

        # Sorrow → meaning, social signals
        sorrow = self.state['sorrow']
        if sorrow > 0.05:
            bias['attention_allocation']['meaning'] = 1.0 + sorrow * 0.5
            bias['memory_access']['reflection'] = 1.0 + sorrow * 0.4

        return bias

    def get_current_state(self):
        return {k: round(v, 3) for k, v in self.state.items()}

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_emotional_valence_priority():
    evp = EmotionalValencePriority()

    print("Initial state:", evp.get_current_state())

    # Simulate an emotional update
    new_input = {
        'joy': 0.8,
        'fear': 0.1,
        'desire': 0.3,
        'sorrow': 0.0
    }

    print("\nUpdating with new emotions:", new_input)
    evp.update(new_input)

    print("Updated state:", evp.get_current_state())

    bias = evp.get_processing_bias()
    print("\nProcessing Biases:")
    for category, adjustments in bias.items():
        print(f"  {category}:")
        for key, value in adjustments.items():
            print(f"    {key}: {round(value, 3)}")

if __name__ == "__main__":
    test_emotional_valence_priority()

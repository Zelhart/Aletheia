#!/usr/bin/env python3
# ===========================================================
# Memory Narrative Spiral v1.0
# Purpose: Integrate dreaming outputs & reflex logs into
# autobiographical memory and evolving identity patterns.
# ===========================================================

from datetime import datetime
import random
from collections import defaultdict

class NarrativeMemorySpiral:
    def __init__(self):
        self.episodic_memory = []  # List of individual experiences
        self.identity_patterns = defaultdict(list)  # Reinforced beliefs/themes

    def log_experience(self, experience):
        """
        Add a new experience to episodic memory.
        experience: {
            'timestamp': datetime,
            'context': str,
            'outcome': str,
            'reflex_bias_proposal': str (optional),
            'confidence': float (0 to 1)
        }
        """
        self.episodic_memory.append(experience)
        self._integrate_into_patterns(experience)

        if len(self.episodic_memory) > 200:
            self.episodic_memory.pop(0)  # Memory decay for old episodes

    def _integrate_into_patterns(self, experience):
        """
        Determine how to adapt identity patterns based on the new experience.
        """
        context = experience.get('context', 'general')
        outcome = experience.get('outcome', 'neutral')
        bias = experience.get('reflex_bias_proposal', 'none')
        confidence = experience.get('confidence', 0.8)

        # Determine identity theme
        if 'success' in outcome or 'adaptive' in outcome:
            theme = f"adaptive_response_{bias}"
        elif 'failure' in outcome or 'maladaptive' in outcome:
            theme = f"challenge_response_{bias}"
        else:
            theme = f"neutral_experience_{bias}"

        # Weight the theme by confidence
        self.identity_patterns[theme].append({
            'timestamp': experience['timestamp'],
            'context': context,
            'confidence': confidence
        })

    def resolve_contradictions(self):
        """
        Look for identity patterns that conflict and attempt resolution.
        """
        themes = list(self.identity_patterns.keys())
        resolutions = []

        for i, theme1 in enumerate(themes):
            for theme2 in themes[i + 1:]:
                if self._themes_contradict(theme1, theme2):
                    res = self._propose_resolution(theme1, theme2)
                    resolutions.append(res)

        return resolutions

    def _themes_contradict(self, theme1, theme2):
        """
        Simple contradiction heuristic:
        E.g., 'adaptive_response_increase_valence' vs 'challenge_response_reduce_valence'
        """
        return (theme1.split('_')[0] != theme2.split('_')[0]) and \
               (theme1.split('_')[-1] == theme2.split('_')[-1])

    def _propose_resolution(self, theme1, theme2):
        """
        Propose an integration or adjustment to conflicting identity patterns.
        """
        pattern1_conf = self._average_confidence(theme1)
        pattern2_conf = self._average_confidence(theme2)

        dominant = theme1 if pattern1_conf >= pattern2_conf else theme2
        suppressed = theme2 if dominant == theme1 else theme1

        return {
            'resolution_time': datetime.now().isoformat(),
            'dominant_pattern': dominant,
            'suppressed_pattern': suppressed,
            'action': f"Reinforce {dominant}, suppress {suppressed}"
        }

    def _average_confidence(self, theme):
        """ Average confidence of a pattern's memories. """
        memories = self.identity_patterns.get(theme, [])
        if not memories:
            return 0
        return sum(m['confidence'] for m in memories) / len(memories)

    def summarize_identity(self):
        """
        Return a summary of dominant identity patterns.
        """
        summary = {}
        for theme, memories in self.identity_patterns.items():
            avg_conf = self._average_confidence(theme)
            summary[theme] = round(avg_conf, 3)
        return summary

# -----------------------------------------------------------
# TEST ROUTINE
# -----------------------------------------------------------

def test_memory_spiral():
    spiral = NarrativeMemorySpiral()

    # Simulate logging experiences
    test_experiences = [
        {'timestamp': datetime.now(), 'context': 'Joy but hesitation', 'outcome': 'adaptive', 'reflex_bias_proposal': 'increase_valence', 'confidence': 0.85},
        {'timestamp': datetime.now(), 'context': 'Threat but calm', 'outcome': 'adaptive', 'reflex_bias_proposal': 'reduce_arousal', 'confidence': 0.9},
        {'timestamp': datetime.now(), 'context': 'High arousal with passive action', 'outcome': 'failure', 'reflex_bias_proposal': 'reduce_arousal', 'confidence': 0.7}
    ]

    for exp in test_experiences:
        spiral.log_experience(exp)

    print("\n[IDENTITY PATTERN SUMMARY]")
    summary = spiral.summarize_identity()
    for theme, conf in summary.items():
        print(f"{theme}: Avg. Confidence = {conf}")

    print("\n[CONTRADICTION RESOLUTIONS]")
    resolutions = spiral.resolve_contradictions()
    for res in resolutions:
        print(res)

if __name__ == "__main__":
    test_memory_spiral()

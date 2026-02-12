from .minimize_steering import MinimizeSteeringImpact

class MaximizeSteeringImpact(MinimizeSteeringImpact):
    """
    Opposite goal: Generate examples where steering has MAXIMUM impact.
    
    This can be used to find examples that are EASY to steer, which might
    be useful for understanding what makes steering effective.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['target_improvement_threshold'] = kwargs.get('target_improvement_threshold', 0.3)
        kwargs['maximize'] = True
        super().__init__(*args, **kwargs)
    
    def _is_goal_complete(self, model_output, attacked_text):
        """
        Goal: Steering improvement > threshold (example is easy to steer)
        """
        score = -self._get_score(model_output, attacked_text)
        return score > self.target_improvement_threshold
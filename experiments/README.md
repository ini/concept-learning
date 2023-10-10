Experiment modules should implement the following functions:

    def make_bottleneck_model(config: dict) -> ConceptBottleneckModel
        """
        Create a bottleneck model from the given configuration.
        """
        ...

    def make_whitening_model(config: dict) -> ConceptWhiteningModel
        """
        Create a whitening model from the given configuration.
        """
        ...
    
    def get_config(**kwargs) -> dict
        """
        Return the default configuration for this experiment
        (optionally overriding with provided keyword arguments).
        """
        ...

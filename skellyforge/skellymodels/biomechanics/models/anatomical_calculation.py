from abc import ABC, abstractmethod
from skellymodels.models.aspect import Aspect
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class CalculationResult:
    """
    Container for results returned by an AnatomicalCalculation.

    Parameters
    ----------
    success : bool
        Whether the calculation completed successfully.
    data : dict
        Dictionary of output values produced by the calculation.
    messages : list of str
        Human-readable logs, warnings, or notes from the computation.
    """
    success: bool
    data: Dict[str, Any]
    messages: List[str]

class AnatomicalCalculation(ABC):
    """
    Abstract base class for anatomical calculations.

    Subclasses must implement both a `calculate()` method to produce
    results from an Aspect, and a `store()` method to write those
    results back into the Aspect.

    Example implementations include:
    - Center of mass estimation
    - Rigid body reconstruction
    - Segment angle derivation
    """
    @abstractmethod
    def calculate(self, aspect:Aspect) -> CalculationResult:
        """
        Perform the calculation using data in the provided Aspect.

        Parameters
        ----------
        aspect : Aspect
            The anatomical data container to operate on.

        Returns
        -------
        CalculationResult
            Success flag, output dictionary, and any log messages.
        """
        pass

    @abstractmethod
    def store(self, aspect:Aspect, results: CalculationResult):
        """
        Save the calculation results into the Aspect object.

        This typically updates `aspect.trajectories` or other
        relevant fields based on the contents of `results.data`.

        Parameters
        ----------
        aspect : Aspect
            The aspect into which to store results.
        results : CalculationResult
            Result object previously produced by `calculate()`.
        """        
        pass
    
    def calculate_and_store(self, aspect:Aspect):
        """
        Convenience method that runs a full calculation pipeline.

        Runs `calculate()`, then calls `store()` if successful. Always
        returns the CalculationResult object for inspection.

        Parameters
        ----------
        aspect : Aspect
            Aspect to process and optionally modify.

        Returns
        -------
        CalculationResult
            Contains outputs and any messages, whether or not the
            calculation succeeded.
        """
        results = self.calculate(aspect)

        if results.success:
            self.store(
                aspect = aspect,
                results=results)

        return results
    
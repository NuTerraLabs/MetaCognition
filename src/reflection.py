"""
Formal Logic Framework for Reflection and Self-Assessment

Implements provability logic-based self-reflection capabilities:
- Confidence predicates
- Uncertainty propagation rules
- Contradiction detection
- Meta-reasoning

Based on modal logic GL (Gödel-Löb logic) and epistemic logic.

Author: Anonymous
Date: January 2026
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Modality(Enum):
    """Modal operators for provability logic"""
    BOX = "□"  # Provable
    DIAMOND = "◇"  # Possibly provable
    CONFIDENCE = "Conf"  # Confidence predicate


@dataclass
class Proposition:
    """
    A logical proposition with associated confidence
    
    Attributes:
        name: Proposition identifier
        confidence: Confidence score in [0, 1]
        evidence: Supporting evidence or internal state
    """
    name: str
    confidence: float
    evidence: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"P({self.name}, u={self.confidence:.3f})"


@dataclass
class MetaProposition:
    """
    A proposition about propositions (meta-level)
    
    Example: "I am confident at level 0.8 that proposition P holds"
    Formal: Conf_0.8(P)
    """
    proposition: Proposition
    modality: Modality
    level: float  # For CONFIDENCE modality
    
    def __repr__(self):
        if self.modality == Modality.CONFIDENCE:
            return f"Conf_{self.level:.2f}({self.proposition.name})"
        else:
            return f"{self.modality.value}{self.proposition.name}"


class ReflectionRules:
    """
    Formal rules for metacognitive reasoning
    
    Implements:
    1. Uncertainty propagation
    2. Contradiction detection
    3. Coherence constraints
    4. Introspection rules
    """
    
    @staticmethod
    def uncertainty_propagation(
        propositions: List[Proposition],
        combination_rule: str = 'min'
    ) -> float:
        """
        Rule 1: Uncertainty Propagation
        
        If Conf_u1(P1), ..., Conf_un(Pn), and P1 ∧ ... ∧ Pn → Q
        Then Conf_f(u1,...,un)(Q)
        
        Args:
            propositions: List of propositions with confidences
            combination_rule: How to combine uncertainties ('min', 'product', 'harmonic')
            
        Returns:
            Combined confidence for conclusion
        """
        if not propositions:
            return 0.0
        
        confidences = [p.confidence for p in propositions]
        
        if combination_rule == 'min':
            # Conservative: weakest link
            return min(confidences)
        elif combination_rule == 'product':
            # Independent events
            return np.prod(confidences)
        elif combination_rule == 'harmonic':
            # Harmonic mean
            return len(confidences) / sum(1/c if c > 0 else float('inf') for c in confidences)
        else:
            raise ValueError(f"Unknown combination rule: {combination_rule}")
    
    @staticmethod
    def detect_contradiction(
        prop_a: Proposition,
        prop_b: Proposition,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Rule 2: Contradiction Detection
        
        If Conf_u(P) and Conf_v(¬P), then Conflict(u, v)
        If u, v > threshold, system recognizes inconsistency
        
        Args:
            prop_a: First proposition
            prop_b: Second proposition (should be negation of first)
            threshold: Minimum confidence to trigger conflict
            
        Returns:
            (is_conflict, conflict_strength)
        """
        if prop_a.confidence > threshold and prop_b.confidence > threshold:
            conflict_strength = min(prop_a.confidence, prop_b.confidence)
            return True, conflict_strength
        return False, 0.0
    
    @staticmethod
    def coherence_constraint(
        positive_confidence: float,
        negative_confidence: float
    ) -> bool:
        """
        Rule 3: Coherence Constraint
        
        Conf_u(P) ∧ Conf_v(¬P) → u + v ≤ 1
        
        A system cannot be highly confident in both P and ¬P
        
        Args:
            positive_confidence: Confidence in P
            negative_confidence: Confidence in ¬P
            
        Returns:
            Whether constraint is satisfied
        """
        return (positive_confidence + negative_confidence) <= 1.0 + 1e-6  # Small tolerance
    
    @staticmethod
    def introspection(prop: Proposition) -> MetaProposition:
        """
        Rule 4: Introspection
        
        Conf_u(P) → □Conf_u(P)
        
        If the system has confidence u in P, it knows it has that confidence
        
        Args:
            prop: Proposition with confidence
            
        Returns:
            Meta-proposition about confidence
        """
        return MetaProposition(
            proposition=prop,
            modality=Modality.BOX,
            level=prop.confidence
        )


class MetacognitiveAgent:
    """
    An agent capable of formal self-reflection
    
    Maintains:
    - Belief base: Set of propositions with confidences
    - Meta-beliefs: Beliefs about beliefs
    - Reflection log: History of metacognitive operations
    """
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.beliefs: Dict[str, Proposition] = {}
        self.meta_beliefs: List[MetaProposition] = []
        self.reflection_log: List[str] = []
    
    def add_belief(
        self,
        name: str,
        confidence: float,
        evidence: Optional[np.ndarray] = None
    ):
        """Add a new belief with confidence"""
        prop = Proposition(name=name, confidence=confidence, evidence=evidence)
        self.beliefs[name] = prop
        self.reflection_log.append(f"Added belief: {prop}")
    
    def reflect_on_belief(self, belief_name: str) -> MetaProposition:
        """
        Reflect on a belief (introspection)
        
        Generates meta-level representation of confidence
        """
        if belief_name not in self.beliefs:
            raise ValueError(f"Belief '{belief_name}' not found")
        
        prop = self.beliefs[belief_name]
        meta_prop = ReflectionRules.introspection(prop)
        self.meta_beliefs.append(meta_prop)
        self.reflection_log.append(f"Reflected on: {belief_name} -> {meta_prop}")
        
        return meta_prop
    
    def check_coherence(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Check coherence of belief system
        
        Returns dictionary of constraint violations
        """
        violations = {}
        
        # Check for contradictions
        belief_names = list(self.beliefs.keys())
        for i, name_a in enumerate(belief_names):
            for name_b in belief_names[i+1:]:
                # Simple check: if names suggest negation
                if name_b.startswith('¬' + name_a) or name_a.startswith('¬' + name_b):
                    prop_a = self.beliefs[name_a]
                    prop_b = self.beliefs[name_b]
                    
                    is_conflict, strength = ReflectionRules.detect_contradiction(
                        prop_a, prop_b
                    )
                    
                    if is_conflict:
                        violations[f"{name_a}_vs_{name_b}"] = False
                        if verbose:
                            print(f"⚠ Contradiction detected: {name_a} (u={prop_a.confidence:.3f}) vs {name_b} (u={prop_b.confidence:.3f})")
                    
                    # Check coherence constraint
                    if not ReflectionRules.coherence_constraint(
                        prop_a.confidence, prop_b.confidence
                    ):
                        violations[f"coherence_{name_a}_{name_b}"] = False
                        if verbose:
                            print(f"⚠ Coherence violation: u({name_a}) + u({name_b}) > 1")
        
        if not violations and verbose:
            print("✓ Belief system is coherent")
        
        return violations
    
    def reason_forward(
        self,
        premises: List[str],
        conclusion_name: str,
        combination_rule: str = 'min'
    ) -> Proposition:
        """
        Forward reasoning with uncertainty propagation
        
        If premises P1, ..., Pn hold, what confidence in conclusion?
        """
        premise_props = []
        for premise_name in premises:
            if premise_name not in self.beliefs:
                raise ValueError(f"Premise '{premise_name}' not found")
            premise_props.append(self.beliefs[premise_name])
        
        # Propagate uncertainty
        conclusion_confidence = ReflectionRules.uncertainty_propagation(
            premise_props, combination_rule
        )
        
        # Create conclusion
        conclusion = Proposition(
            name=conclusion_name,
            confidence=conclusion_confidence
        )
        
        self.beliefs[conclusion_name] = conclusion
        self.reflection_log.append(
            f"Inferred: {premises} -> {conclusion_name} (u={conclusion_confidence:.3f})"
        )
        
        return conclusion
    
    def assess_uncertainty(self) -> Dict[str, float]:
        """
        Assess overall uncertainty in belief system
        
        Returns statistics about confidence distribution
        """
        if not self.beliefs:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        confidences = [b.confidence for b in self.beliefs.values()]
        
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'entropy': -np.sum([c * np.log(c + 1e-10) + (1-c) * np.log(1-c + 1e-10) 
                               for c in confidences]) / len(confidences)
        }
    
    def explain_belief(self, belief_name: str) -> str:
        """Generate explanation for a belief"""
        if belief_name not in self.beliefs:
            return f"No belief named '{belief_name}'"
        
        prop = self.beliefs[belief_name]
        
        explanation = f"Belief: {belief_name}\n"
        explanation += f"Confidence: {prop.confidence:.3f}\n"
        
        if prop.confidence > 0.8:
            explanation += "Assessment: High confidence - reliable\n"
        elif prop.confidence > 0.5:
            explanation += "Assessment: Moderate confidence - uncertain\n"
        else:
            explanation += "Assessment: Low confidence - unreliable\n"
        
        # Check for related meta-beliefs
        related_meta = [mb for mb in self.meta_beliefs 
                       if mb.proposition.name == belief_name]
        if related_meta:
            explanation += f"Meta-reflections: {len(related_meta)}\n"
        
        return explanation
    
    def print_state(self):
        """Print current state of agent's beliefs"""
        print(f"\n{'='*60}")
        print(f"Agent: {self.name}")
        print(f"{'='*60}")
        print(f"Beliefs: {len(self.beliefs)}")
        for name, prop in self.beliefs.items():
            print(f"  • {name}: u={prop.confidence:.3f}")
        
        print(f"\nMeta-beliefs: {len(self.meta_beliefs)}")
        for mb in self.meta_beliefs[:5]:  # Show first 5
            print(f"  • {mb}")
        
        stats = self.assess_uncertainty()
        print(f"\nUncertainty Statistics:")
        print(f"  Mean confidence: {stats['mean']:.3f}")
        print(f"  Std deviation: {stats['std']:.3f}")
        print(f"  Entropy: {stats['entropy']:.3f}")
        print(f"{'='*60}\n")


def demonstrate_reflection_system():
    """Demonstration of the formal reflection system"""
    print("="*80)
    print("DEMONSTRATION: Formal Metacognitive Reflection System")
    print("="*80)
    
    # Create agent
    agent = MetacognitiveAgent(name="Socrates")
    
    # Add beliefs
    print("\n1. Adding beliefs...")
    agent.add_belief("sky_is_blue", confidence=0.95)
    agent.add_belief("grass_is_green", confidence=0.90)
    agent.add_belief("uncertain_fact", confidence=0.45)
    agent.add_belief("conflicting_fact", confidence=0.60)
    agent.add_belief("¬conflicting_fact", confidence=0.55)  # Contradiction!
    
    # Print state
    agent.print_state()
    
    # Reflect on beliefs
    print("\n2. Reflecting on beliefs (introspection)...")
    agent.reflect_on_belief("sky_is_blue")
    agent.reflect_on_belief("uncertain_fact")
    
    # Check coherence
    print("\n3. Checking belief coherence...")
    agent.check_coherence(verbose=True)
    
    # Forward reasoning
    print("\n4. Forward reasoning...")
    agent.reason_forward(
        premises=["sky_is_blue", "grass_is_green"],
        conclusion_name="natural_colors_exist",
        combination_rule='min'
    )
    
    # Explain belief
    print("\n5. Explaining beliefs...")
    print(agent.explain_belief("natural_colors_exist"))
    print(agent.explain_belief("uncertain_fact"))
    
    # Final state
    agent.print_state()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demonstrate_reflection_system()

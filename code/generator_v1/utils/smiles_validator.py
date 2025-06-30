"""
SMILES validation utilities for ensuring chemically valid and organic compounds.
"""
import re
from typing import List, Set, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen


class SMILESValidator:
    """Validates SMILES for chemical validity and organic content."""
    
    def __init__(self):
        """Initialize validator with organic element constraints."""
        # Common organic elements (allowing for drug-like compounds)
        self.organic_elements = {
            'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B'
        }
        
        # Patterns for clearly non-organic elements
        self.inorganic_patterns = [
            # Metals
            r'[A-Z][a-z]?\+\d*',  # Metal ions like Tl+3, Pu, etc.
            r'\b(Tc|Tl|Pu|Ra|Ac|Th|Pa|U|Np|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)\b',  # Actinides/lanthanides
            r'\b(Li|Na|K|Rb|Cs|Fr|Be|Mg|Ca|Sr|Ba|Ra)\b',  # Alkali/alkaline earth metals
            r'\b(Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd)\b',  # Transition metals
            r'\b(Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Al|Ga|In|Tl|Sn|Pb|Bi)\b',  # Other metals
        ]
        
        # Compile patterns for efficiency
        self.inorganic_regex = re.compile('|'.join(self.inorganic_patterns), re.IGNORECASE)
        
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is chemically valid using RDKit."""
        if not smiles or len(smiles.strip()) == 0:
            return False
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
    
    def is_organic(self, smiles: str) -> bool:
        """Check if SMILES represents an organic compound."""
        if not self.is_valid_smiles(smiles):
            return False
            
        # Quick pattern check for obviously inorganic elements
        if self.inorganic_regex.search(smiles):
            return False
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # Check all atoms in the molecule
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in self.organic_elements:
                    return False
                    
            return True
        except Exception:
            return False
    
    def is_drug_like(self, smiles: str) -> bool:
        """Check if SMILES satisfies basic drug-likeness criteria (Lipinski's Rule of Five)."""
        if not self.is_organic(smiles):
            return False
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Standard Lipinski criteria
            if mw > 500:
                return False
            if logp > 5:
                return False
            if hbd > 5:
                return False
            if hba > 10:
                return False
                
            return True
        except Exception:
            return False
    
    def filter_valid_organic(self, smiles_list: List[str]) -> List[Tuple[int, str]]:
        """Filter list of SMILES to only valid organic compounds."""
        valid_smiles = []
        for i, smiles in enumerate(smiles_list):
            if self.is_organic(smiles):
                valid_smiles.append((i, smiles))
        return valid_smiles
    
    def get_validation_stats(self, smiles_list: List[str]) -> dict:
        """Get validation statistics for a list of SMILES."""
        total = len(smiles_list)
        valid_count = 0
        organic_count = 0
        drug_like_count = 0
        
        for smiles in smiles_list:
            if self.is_valid_smiles(smiles):
                valid_count += 1
                if self.is_organic(smiles):
                    organic_count += 1
                    if self.is_drug_like(smiles):
                        drug_like_count += 1
        
        return {
            'total': total,
            'valid': valid_count,
            'organic': organic_count,
            'drug_like': drug_like_count,
            'valid_percent': (valid_count / total * 100) if total > 0 else 0,
            'organic_percent': (organic_count / total * 100) if total > 0 else 0,
            'drug_like_percent': (drug_like_count / total * 100) if total > 0 else 0,
        }


def test_validator():
    """Test the SMILES validator."""
    validator = SMILESValidator()
    
    # Test cases
    test_smiles = [
        "CCO",  # Ethanol - valid organic
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - valid organic drug
        "Tl+3",  # Thallium ion - invalid inorganic
        "Pu",   # Plutonium - invalid inorganic
        "Tc+5", # Technetium - invalid inorganic
        "",     # Empty string
        "INVALID",  # Invalid SMILES
        "C1=CC=CC=C1",  # Benzene - valid organic
    ]
    
    print("SMILES Validation Test:")
    for smiles in test_smiles:
        valid = validator.is_valid_smiles(smiles)
        organic = validator.is_organic(smiles)
        drug_like = validator.is_drug_like(smiles)
        print(f"'{smiles}': Valid={valid}, Organic={organic}, Drug-like={drug_like}")
    
    # Test statistics
    stats = validator.get_validation_stats(test_smiles)
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    test_validator()

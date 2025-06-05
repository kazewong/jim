import enum
from jimgw.run.library.IMRPhenomPv2_standard_cbc import IMRPhenomPv2StandardCBCRunDefinition

class AvailableDefinitions(enum.Enum):
    """
    Enum for available run definitions.
    """
    IMRPHENOMPV2STANDARDCBC = IMRPhenomPv2StandardCBCRunDefinition

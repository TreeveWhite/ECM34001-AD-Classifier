"""
exceptions.py
==============================================
A custom exception was defined by this project to be raised when no relevent axial
slices are found within an MRI scan. A customexception was used by this project to do
this so it could be caught and excluded from other exceptions. 
"""


class NoGoodSlicesException(Exception):
    """
    No Good Slices Excepion
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

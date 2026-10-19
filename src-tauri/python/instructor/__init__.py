"""The Instructor — an adapter that turns a one-line brief into the training
set for a new adapter.

The rule the whole package is built around: **the model plans, code executes.**
The Instructor emits a fixed-shape JSON plan and nothing else. Every file that
ends up on disk is written by Python, from a plan that passed validation.
"""

__version__ = "0.1.0"

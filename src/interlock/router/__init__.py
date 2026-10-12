"""Advisory router. The policy is a pure function. The command line does the reading."""

from interlock.router.policy import Decision, Refusal, Request, decide

__all__ = ["Decision", "Refusal", "Request", "decide"]

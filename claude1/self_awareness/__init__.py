"""Self-awareness module — introspection, codebase index, factories, and self-modification."""

from claude1.self_awareness.introspection import CodebaseIntrospector
from claude1.self_awareness.codebase_index import CodebaseIndex
from claude1.self_awareness.skill_factory import SkillFactory
from claude1.self_awareness.role_factory import RoleFactory
from claude1.self_awareness.self_modifier import SelfModifier

__all__ = [
    "CodebaseIntrospector",
    "CodebaseIndex",
    "SkillFactory",
    "RoleFactory",
    "SelfModifier",
]

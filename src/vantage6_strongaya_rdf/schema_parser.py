"""
------------------------------------------------------------------------------
Schema Parser Module

Provides functionality to parse the AYA cancer JSON-LD schema and build
dynamic SPARQL predicate paths based on schemaReconstruction.
------------------------------------------------------------------------------
"""

from typing import Dict, FrozenSet, List, Optional, Set
from vantage6_strongaya_general.miscellaneous import safe_log

# JSON-LD node type that represents a class in a schemaReconstruction
CLASS_NODE_TYPE = "schema:ClassNode"

# Placement of a schemaReconstruction node relative to the variable's own node
PLACEMENT_AFTER = "after"

# Variable that describes the patient's identifier, and its JSON-LD type
IDENTIFIER_VARIABLE_NAME = "identifier"
IDENTIFIER_VARIABLE_TYPE = "schema:IdentifierVariable"

# Identifier predicate and class that are used when the schema describes neither
DEFAULT_IDENTIFIER_PREDICATE = "sio:SIO_000673"
DEFAULT_IDENTIFIER_CLASS = "ncit:C25364"


def get_schema_prefixes(schema: dict) -> Dict[str, str]:
    """
    Extract all prefix mappings from the JSON-LD schema.

    Prefixes are extracted from two locations:
    1. The @context section (top-level JSON-LD context)
    2. The schema.prefixes section (schema-specific prefix definitions)

    When the same prefix appears in both locations, the value from
    schema.prefixes takes precedence.

    Args:
        schema: The full schema dictionary loaded from JSON-LD

    Returns:
        Dictionary mapping prefix names (without colon) to their URI values.
        Example: {"ncit": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#", ...}
    """
    prefixes = {}

    # Extract from @context (JSON-LD standard location)
    context = schema.get("@context", {})
    for key, value in context.items():
        # Skip JSON-LD special keys
        if key.startswith("@"):
            continue
        # Only include string values that look like absolute URIs.
        # This excludes internal relative-path shorthand entries (e.g. "schema": "schema/")
        # that are used for JSON-LD structural purposes rather than as ontology prefixes.
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            prefixes[key] = value

    # Extract from schema.prefixes (schema-specific location)
    schema_prefixes = schema.get("schema", {}).get("prefixes", {})
    for key, value in schema_prefixes.items():
        # Only include string values that look like absolute URIs
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            prefixes[key] = value

    safe_log("info", f"Extracted {len(prefixes)} prefixes from schema")
    return prefixes


def _get_variables(schema: dict) -> dict:
    """
    Retrieve the variable definitions of the schema.

    Args:
        schema: The full schema dictionary

    Returns:
        Dictionary of variable definitions, empty when the schema holds none
    """
    return schema.get("schema", {}).get("variables", {}) or {}


def _get_reconstruction(variable_definition: dict) -> List[dict]:
    """
    Retrieve the schemaReconstruction of a single variable definition.

    Args:
        variable_definition: Definition of a single schema variable

    Returns:
        List of schemaReconstruction nodes, empty when the variable holds none
    """
    return variable_definition.get("schemaReconstruction", []) or []


def get_intermediate_classes(schema: dict) -> Set[str]:
    """
    Determine the intermediate (container) classes of the schema.

    Intermediate classes - such as PROM, EHR and HCPROM - are not reached from the
    patient directly; they are attached to the node of another variable and are
    therefore declared with an "after" placement by the variables that point to them.

    Args:
        schema: The full schema dictionary

    Returns:
        Set of class codes that are used as intermediate class (e.g. {"ncit:C177377"})
    """
    return {
        item.get("class")
        for variable_definition in _get_variables(schema).values()
        for item in _get_reconstruction(variable_definition)
        if item.get("placement") == PLACEMENT_AFTER and item.get("class")
    }


def _as_alternation(alternatives: List[str]) -> str:
    """
    Combine path alternatives into a single SPARQL property path segment.

    Args:
        alternatives: Alternative (sequence) paths, e.g. ["sio:SIO_000255/sio:SIO_000008"]

    Returns:
        The single alternative itself, or a grouped alternation such as "(a/b|a/b/b)"
    """
    if len(alternatives) == 1:
        return alternatives[0]
    return "(" + "|".join(alternatives) + ")"


def _build_class_node_segments(
    variable_definition: dict, schema: dict, resolving: FrozenSet[str]
) -> List[str]:
    """
    Build the ordered path segments that precede a variable's own predicate.

    Args:
        variable_definition: Definition of a single schema variable
        schema: The full schema dictionary
        resolving: Intermediate classes that are currently being resolved (loop guard)

    Returns:
        Ordered list of SPARQL property path segments
    """
    intermediate_classes = get_intermediate_classes(schema)
    segments: List[str] = []

    for item in _get_reconstruction(variable_definition):
        # Only class nodes that precede the variable's own node contribute to the path;
        # "after" nodes are reached from the variable's node and unit nodes hold no path
        if (
            item.get("@type") != CLASS_NODE_TYPE
            or item.get("placement") == PLACEMENT_AFTER
        ):
            continue

        item_class = item.get("class")
        item_predicate = item.get("predicate")

        if item_class in intermediate_classes:
            intermediate_segments = resolve_intermediate_class_path(
                item_class, schema, hop_predicate=item_predicate, resolving=resolving
            )
            if intermediate_segments:
                segments.extend(intermediate_segments)
                continue
            safe_log(
                "warning",
                f"Could not resolve the path to intermediate class '{item_class}'; "
                f"using its own predicate instead",
            )

        if item_predicate:
            segments.append(item_predicate)

    return segments


def resolve_intermediate_class_path(
    target_class: str,
    schema: dict,
    hop_predicate: Optional[str] = None,
    resolving: FrozenSet[str] = frozenset(),
) -> List[str]:
    """
    Resolve the predicate path that reaches an intermediate class (like PROM, EHR, HCPROM).

    Intermediate classes are attached to the node of another variable, which is why they
    are declared with an "after" placement by the variables that point to them. The path
    to such a class therefore consists of two parts:
    1. The path to the node of a variable that points to the intermediate class; and
    2. The hop from that variable's node to the intermediate class itself.

    Variables may reach the intermediate class through different routes, in which case
    all distinct routes are combined into a single grouped alternation. The resolution is
    recursive, so intermediate classes that are themselves nested are resolved as well.

    Args:
        target_class: The class to resolve (e.g., "ncit:C177377" for PROM)
        schema: The full schema dictionary
        hop_predicate: The predicate that the requesting variable declares for the hop
                       into the intermediate class; combined with the predicates that
                       the schema declares for that hop when the two differ
        resolving: Intermediate classes that are currently being resolved (loop guard)

    Returns:
        Ordered list of SPARQL property path segments that reach the intermediate class,
        or an empty list when the class is not used as an intermediate class
    """
    if target_class in resolving:
        safe_log(
            "warning",
            f"Circular intermediate class definition detected for '{target_class}'",
        )
        return []

    resolving = resolving | {target_class}

    routes: Set[str] = set()
    hop_predicates: Set[str] = set()

    for variable_definition in _get_variables(schema).values():
        for item in _get_reconstruction(variable_definition):
            if (
                item.get("placement") != PLACEMENT_AFTER
                or item.get("class") != target_class
            ):
                continue

            # Path to the node of the variable that points to the intermediate class
            route = _build_class_node_segments(variable_definition, schema, resolving)
            own_predicate = variable_definition.get("predicate")
            if own_predicate:
                route = route + [own_predicate]
            if route:
                routes.add("/".join(route))

            # Hop from that variable's node to the intermediate class
            if item.get("predicate"):
                hop_predicates.add(item["predicate"])

    if not routes:
        return []

    if hop_predicate and hop_predicate not in hop_predicates:
        if hop_predicates:
            safe_log(
                "warning",
                f"Predicate '{hop_predicate}' declared for intermediate class "
                f"'{target_class}' differs from the schema's "
                f"{sorted(hop_predicates)}; both are considered",
            )
        hop_predicates.add(hop_predicate)

    segments = [_as_alternation(sorted(routes))]
    if hop_predicates:
        segments.append(_as_alternation(sorted(hop_predicates)))

    return segments


def build_predicate_path(variable_name: str, schema: dict) -> str:
    """
    Build the SPARQL predicate path from schema definition.

    Rules:
    1. Collect the predicates of the class nodes that precede the variable's own node
    2. Resolve intermediate classes (like PROM, EHR and HCPROM) to their full path
    3. Append the variable's own predicate

    The predicates are combined into an ordered sequence path, which follows the exact
    structure that the schema describes rather than any combination of its predicates.

    Args:
        variable_name: Name of the variable to build path for
        schema: The full schema dictionary

    Returns:
        SPARQL property path string like "sio:SIO_000255/sio:SIO_000008"
    """
    variables = _get_variables(schema)

    if variable_name not in variables:
        safe_log("warning", f"Variable '{variable_name}' not found in schema")
        return ""

    variable_definition = variables[variable_name]

    segments = _build_class_node_segments(variable_definition, schema, frozenset())

    # Add the main predicate of the variable itself
    main_predicate = variable_definition.get("predicate")
    if main_predicate:
        segments.append(main_predicate)

    if not segments:
        safe_log("warning", f"No predicates found for variable '{variable_name}'")
        return ""

    # Format: predicate1/predicate2
    path = "/".join(segments)

    safe_log("info", f"Built predicate path for '{variable_name}': {path}")
    return path


def _resolve_variable_name(variable_name: str, schema: dict) -> str:
    """
    Resolve a variable identifier to its schema variable name.

    Supports lookup by:
    1. Direct variable name (e.g., "biological_sex")
    2. Class code (e.g., "ncit:C28421") - reverse lookup by class

    Args:
        variable_name: Variable name or class code
        schema: The full schema dictionary

    Returns:
        The resolved schema variable name, or empty string if not found
    """
    variables = _get_variables(schema)

    # Direct name lookup
    if variable_name in variables:
        return variable_name

    # Reverse lookup by class code
    for var_name, var_def in variables.items():
        if var_def.get("class") == variable_name:
            safe_log(
                "info",
                f"Resolved class code '{variable_name}' to variable '{var_name}'",
            )
            return var_name

    return ""


def get_identifier_query_params(schema: dict) -> dict:
    """
    Get the query parameters of the patient's identifier.

    The schema describes the identifier just like any other variable, which means that
    the predicate and class that a query needs to resolve a patient's identifier can be
    derived from it rather than being hardcoded in the query templates. The defaults are
    only used when the schema describes no identifier at all.

    Args:
        schema: The full schema dictionary

    Returns:
        Dictionary with:
        {
            "predicate": "sio:SIO_000673",
            "class": "ncit:C25364"
        }
    """
    variables = _get_variables(schema)

    identifier_name = ""
    if IDENTIFIER_VARIABLE_NAME in variables:
        identifier_name = IDENTIFIER_VARIABLE_NAME
    else:
        # Fall back to the first variable that is typed as an identifier
        for var_name, var_def in variables.items():
            if (
                var_def.get("@type") == IDENTIFIER_VARIABLE_TYPE
                or var_def.get("dataType") == "identifier"
            ):
                identifier_name = var_name
                break

    if not identifier_name:
        safe_log(
            "warning",
            "Schema does not describe an identifier variable; "
            f"using '{DEFAULT_IDENTIFIER_PREDICATE}' and "
            f"'{DEFAULT_IDENTIFIER_CLASS}' instead",
        )
        return {
            "predicate": DEFAULT_IDENTIFIER_PREDICATE,
            "class": DEFAULT_IDENTIFIER_CLASS,
        }

    # The identifier is described without a schemaReconstruction, which means that its
    # path is the bare predicate; a reconstruction is honoured should one be added
    predicate = (
        build_predicate_path(identifier_name, schema) or DEFAULT_IDENTIFIER_PREDICATE
    )
    identifier_class = (
        variables[identifier_name].get("class") or DEFAULT_IDENTIFIER_CLASS
    )

    return {"predicate": predicate, "class": identifier_class}


def get_variable_query_params(variable_name: str, schema: dict) -> dict:
    """
    Get all query parameters for a variable.

    Supports lookup by variable name or class code (e.g., "ncit:C28421").

    Args:
        variable_name: Name of the variable or class code
        schema: The full schema dictionary

    Returns:
        Dictionary with:
        {
            "predicate_path": "sio:SIO_000255/sio:SIO_000008",
            "main_class": "ncit:C156420",
            "ontology_prefix": "ncit:"
        }
    """
    resolved_name = _resolve_variable_name(variable_name, schema)

    if not resolved_name:
        safe_log(
            "warning",
            f"Variable '{variable_name}' not found in schema by name or class",
        )
        return {}

    variables = _get_variables(schema)
    var_def = variables[resolved_name]

    # Get main class
    main_class = var_def.get("class", "")

    # Extract ontology prefix
    ontology_prefix = ""
    if ":" in main_class:
        ontology_prefix = main_class.split(":")[0] + ":"

    # Build predicate path
    predicate_path = build_predicate_path(resolved_name, schema)

    return {
        "predicate_path": predicate_path,
        "main_class": main_class,
        "ontology_prefix": ontology_prefix,
    }

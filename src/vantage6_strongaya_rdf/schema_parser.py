"""
------------------------------------------------------------------------------
Schema Parser Module

Provides functionality to parse the AYA cancer JSON-LD schema and build 
dynamic SPARQL predicate paths based on schemaReconstruction.
------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional
from vantage6_strongaya_general.miscellaneous import safe_log


def resolve_intermediate_class_path(target_class: str, schema: dict) -> List[str]:
    """
    Find the predicate path to reach an intermediate class (like PROM, EHR, HCPROM)
    by looking at how other variables connect to it via "after" placement.
    
    Args:
        target_class: The class to resolve (e.g., "ncit:C177377" for PROM)
        schema: The full schema dictionary
        
    Returns:
        List of predicates that form the path to the intermediate class
    """
    variables = schema.get('schema', {}).get('variables', {})
    
    # Search for a variable that has this class as an "after" placement
    for var_name, var_def in variables.items():
        schema_rec = var_def.get('schemaReconstruction', [])
        
        for item in schema_rec:
            # Check if this is an "after" placement pointing to our target class
            if item.get('placement') == 'after' and item.get('class') == target_class:
                # Found a variable that connects to this class
                # Build the path for that variable and return it
                predicates = []
                
                # Collect "before" predicates (or default if no placement)
                for rec_item in schema_rec:
                    placement = rec_item.get('placement')
                    # Only include items that come "before" (or have no placement, which defaults to "before")
                    # Skip "after" items and non-ClassNode items (like UnitNode)
                    if rec_item.get('@type') == 'schema:ClassNode':
                        if placement != 'after':
                            predicates.append(rec_item.get('predicate'))
                
                # Add the main predicate of the variable
                main_predicate = var_def.get('predicate')
                if main_predicate:
                    predicates.append(main_predicate)
                
                return predicates
    
    # If not found, return empty list
    return []


def build_predicate_path(variable_name: str, schema: dict) -> str:
    """
    Build the SPARQL predicate path from schema definition.
    
    Rules:
    1. Collect "before" predicates from schemaReconstruction
    2. Add the main predicate
    3. If a "before" class is an intermediate class (like PROM, EHR),
       recursively resolve its full path from other variables
    
    Args:
        variable_name: Name of the variable to build path for
        schema: The full schema dictionary
        
    Returns:
        SPARQL property path string like "(sio:SIO_000255|sio:SIO_000008)*"
    """
    variables = schema.get('schema', {}).get('variables', {})
    
    if variable_name not in variables:
        safe_log("warning", f"Variable '{variable_name}' not found in schema")
        return ""
    
    var_def = variables[variable_name]
    schema_rec = var_def.get('schemaReconstruction', [])
    
    predicates = []
    
    # Collect "before" predicates (and resolve intermediate classes)
    for item in schema_rec:
        # Only process ClassNode items with "before" placement (or no placement, which defaults to "before")
        if item.get('@type') == 'schema:ClassNode':
            placement = item.get('placement')
            
            if placement != 'after':  # "before" or default (None)
                item_class = item.get('class')
                item_predicate = item.get('predicate')
                
                # Check if this class is an intermediate class that needs resolution
                # Intermediate classes are those that other variables connect to via "after"
                intermediate_path = resolve_intermediate_class_path(item_class, schema)
                
                if intermediate_path:
                    # This is an intermediate class, use its resolved path
                    predicates.extend(intermediate_path)
                else:
                    # Regular "before" predicate
                    if item_predicate:
                        predicates.append(item_predicate)
    
    # Add the main predicate of the variable itself
    main_predicate = var_def.get('predicate')
    if main_predicate:
        predicates.append(main_predicate)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_predicates = []
    for pred in predicates:
        if pred not in seen:
            seen.add(pred)
            unique_predicates.append(pred)
    
    # Build the SPARQL property path
    if not unique_predicates:
        safe_log("warning", f"No predicates found for variable '{variable_name}'")
        return ""
    
    # Format: (predicate1|predicate2|predicate3)*
    path = "(" + "|".join(unique_predicates) + ")*"
    
    safe_log("info", f"Built predicate path for '{variable_name}': {path}")
    return path


def get_variable_query_params(variable_name: str, schema: dict) -> dict:
    """
    Get all query parameters for a variable.
    
    Args:
        variable_name: Name of the variable
        schema: The full schema dictionary
        
    Returns:
        Dictionary with:
        {
            "predicate_path": "(sio:SIO_000255|sio:SIO_000008)*",
            "main_class": "ncit:C156420",
            "ontology_prefix": "ncit:"
        }
    """
    variables = schema.get('schema', {}).get('variables', {})
    
    if variable_name not in variables:
        safe_log("error", f"Variable '{variable_name}' not found in schema")
        return {}
    
    var_def = variables[variable_name]
    
    # Get main class
    main_class = var_def.get('class', '')
    
    # Extract ontology prefix
    ontology_prefix = ""
    if ':' in main_class:
        ontology_prefix = main_class.split(':')[0] + ':'
    
    # Build predicate path
    predicate_path = build_predicate_path(variable_name, schema)
    
    return {
        "predicate_path": predicate_path,
        "main_class": main_class,
        "ontology_prefix": ontology_prefix
    }

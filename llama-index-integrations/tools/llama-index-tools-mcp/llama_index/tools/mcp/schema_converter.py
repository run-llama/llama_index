"""Utilities for converting JSON Schema to Pydantic models."""

from typing import Any, Dict, List, Union, Literal, Type, TYPE_CHECKING
from pydantic import Field, create_model

# Map JSON Schema types to Python types
json_type_mapping: Dict[str, Type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": Dict,
    "array": List,
}


class TypeResolutionMixin:
    """Mixin for resolving JSON Schema types to Python types."""
    
    def _resolve_field_type(
        self: "SchemaConverter",
        field_schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve the Python type from a field schema."""
        if "$ref" in field_schema:
            return self._resolve_reference(field_schema, defs)
        if "enum" in field_schema:
            return Literal[tuple(field_schema["enum"])]
        if "anyOf" in field_schema:
            return self._resolve_union_type(field_schema, defs)
        return self._resolve_basic_type(field_schema, defs)

    def _resolve_reference(
        self: "SchemaConverter",
        field_schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a $ref reference."""
        ref_name = self._extract_ref_name(field_schema["$ref"])

        if ref_name not in defs:
            # This is where we would have used self.properties_cache
            # For standalone usage, we'll just return the reference name as string or handle appropriately
            return str

        ref_schema = defs[ref_name]

        if "anyOf" in ref_schema:
            return self._resolve_union_type(ref_schema, defs)
        if self._is_simple_array(ref_schema):
            return self._create_list_type(ref_schema, defs)
        if self._is_simple_object(ref_schema):
            return self._create_dict_type(ref_schema, defs)
        # For now, return a simple dict type as fallback
        return Dict[str, Any]

    def _resolve_union_type(
        self: "SchemaConverter",
        schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a Union type (anyOf)."""
        union_types = [
            self._resolve_union_option(option, defs) for option in schema["anyOf"]
        ]
        return Union[tuple(union_types)] if len(union_types) > 1 else union_types[0]

    def _resolve_union_option(
        self: "SchemaConverter",
        option: dict,
        defs: dict,
    ) -> Any:
        """Resolve a single option in a union type."""
        if "$ref" in option:
            return self._resolve_reference(option, defs)
        if "enum" in option:
            return Literal[tuple(option["enum"])]
        if option.get("type") == "null":
            return type(None)
        return self._resolve_basic_type(option, defs)

    def _resolve_basic_type(
        self: "SchemaConverter",
        schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a basic JSON Schema type."""
        json_type = schema.get("type", "string")
        json_type = json_type[0] if isinstance(json_type, list) else json_type

        if self._is_simple_array(schema):
            return self._create_list_type(schema, defs)
        if self._is_simple_object(schema):
            return self._create_dict_type(schema, defs)
        return json_type_mapping.get(json_type, str)

    def _is_simple_array(self: "SchemaConverter", schema: dict) -> bool:
        """Check if schema is a simple array type."""
        return schema.get("type") == "array" and "items" in schema

    def _is_simple_object(self: "SchemaConverter", schema: dict) -> bool:
        """Check if schema is a simple object type."""
        additional_props = schema.get("additionalProperties")
        return (
            schema.get("type") == "object"
            and "additionalProperties" in schema
            and additional_props is not False
            and isinstance(additional_props, dict)
        )

    def _extract_ref_name(self: "SchemaConverter", ref_path: str) -> str:
        """Extract reference name from $ref path."""
        return ref_path.split("#/$defs/")[-1]


class TypeCreationMixin:
    """Mixin for creating Python types from schema definitions."""
    
    def _create_list_type(self: "SchemaConverter", schema: dict, defs: dict) -> type:
        """Create a List type from schema."""
        item_type = self._resolve_field_type(schema["items"], defs)
        return List[item_type]

    def _create_dict_type(self: "SchemaConverter", schema: dict, defs: dict) -> type:
        """Create a Dict type from schema."""
        additional_props = schema.get("additionalProperties")

        if additional_props is False or additional_props is None:
            return Dict[str, Any]

        if isinstance(additional_props, dict):
            value_type = self._resolve_field_type(additional_props, defs)
            return Dict[str, value_type]

        return Dict[str, Any]

    def _is_simple_array(self: "SchemaConverter", schema: dict) -> bool:
        """Check if schema is a simple array type."""
        return schema.get("type") == "array" and "items" in schema

    def _is_simple_object(self: "SchemaConverter", schema: dict) -> bool:
        """Check if schema is a simple object type."""
        additional_props = schema.get("additionalProperties")
        return (
            schema.get("type") == "object"
            and "additionalProperties" in schema
            and additional_props is not False
            and isinstance(additional_props, dict)
        )

    def _extract_ref_name(self: "SchemaConverter", ref_path: str) -> str:
        """Extract reference name from $ref path."""
        return ref_path.split("#/$defs/")[-1]


class FieldExtractionMixin:
    """Mixin for extracting fields from schema definitions."""

    def _extract_fields(self: "SchemaConverter", schema: dict, defs: dict) -> dict:
        """Extract Pydantic fields from schema."""
        properties = self._get_properties(schema)
        required_fields = set(schema.get("required", []))

        # For enum schemas, treat them as required by default
        if "enum" in schema:
            required_fields = {schema.get("title", "enum_field")}

        fields = {}
        for field_name, field_schema in properties.items():
            field_type = self._resolve_field_type(field_schema, defs)
            default_value, final_type = self._set_field_default(
                field_name,
                required_fields,
                field_type,
                field_schema,
            )

            fields[field_name] = (
                final_type,
                Field(default_value, description=field_schema.get("description", "")),
            )

        return fields

    def _get_properties(self: "SchemaConverter", schema: dict) -> dict:
        """Get properties from schema, handling enum types."""
        if "enum" in schema:
            # For enum types, create a property with the schema name as the key
            # This ensures the enum is treated as a required field
            return {schema.get("title", "enum_field"): schema}
        return schema.get("properties", {})

    @staticmethod
    def _set_field_default(
        field: str,
        required_fields: set[str],
        ftype: Any,
        field_schema: dict,
    ) -> tuple[type(Ellipsis) | None, Any]:
        """Set default value and make type optional if needed."""
        if field in required_fields:
            return ..., ftype
        default_value = field_schema.get("default")
        if default_value is None:
            ftype = ftype | type(None)
        return default_value, ftype


class SchemaConverter(TypeResolutionMixin, TypeCreationMixin, FieldExtractionMixin):
    """A standalone converter from JSON Schema to Pydantic models."""
    
    def __init__(self):
        self.properties_cache = {}
        
    def create_model_from_json_schema(
        self,
        schema: dict[str, Any],
        model_name: str,
        defs: dict[str, Any] = None
    ) -> type:
        """
        Create a Pydantic model from a JSON Schema.
        
        Args:
            schema: The JSON Schema to convert
            model_name: The name for the resulting Pydantic model
            defs: Definitions referenced in the schema
            
        Returns:
            A Pydantic model class
        """
        if defs is None:
            defs = {}
            
        fields = self._extract_fields(schema, defs)
        
        # Create and return the Pydantic model
        return create_model(model_name, **fields)

    def remove_model_fields(
        self,
        model: type,
        fields_to_remove: set[str],
        model_name: str
    ) -> type:
        """
        Remove specified fields from a Pydantic model.
        
        Args:
            model: The Pydantic model to modify
            fields_to_remove: Set of field names to remove
            model_name: The name for the resulting Pydantic model
            
        Returns:
            A new Pydantic model class with specified fields removed
        """
        # Get current model fields
        model_fields = model.__fields__
        
        # Filter out the fields to remove
        new_fields = {}
        for field_name, field in model_fields.items():
            if field_name not in fields_to_remove:
                new_fields[field_name] = (field.type_, field.field_info)
                
        # Create new model without specified fields
        return create_model(model_name, **new_fields)

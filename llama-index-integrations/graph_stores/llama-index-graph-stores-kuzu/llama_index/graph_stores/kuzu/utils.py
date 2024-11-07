from typing import List, _LiteralGenericAlias, get_args, Tuple
import kuzu

Triple = Tuple[str, str, str]


def create_fresh_database(db: str) -> None:
    """
    Create a new Kùzu database by removing existing database directory and its contents.
    """
    import shutil

    shutil.rmtree(db, ignore_errors=True)


def get_list_from_literal(literal: _LiteralGenericAlias) -> List[str]:
    """
    Get a list of strings from a Literal type.

    Parameters:
    literal (_LiteralGenericAlias): The Literal type from which to extract the strings.

    Returns:
    List[str]: A list of strings extracted from the Literal type.
    """
    if not isinstance(literal, _LiteralGenericAlias):
        raise TypeError(
            f"{literal} must be a Literal type.\nTry using typing.Literal{literal}."
        )
    return list(get_args(literal))


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values and remove the `e.` prefix from the keys
    return {key.replace("e.", ""): value for key, value in input_dict.items() if value}


def get_filtered_props(records: dict, filter_list: List[str]) -> dict:
    return {k: v for k, v in records.items() if k not in filter_list}


# Lookup entry by middle value of tuple
def lookup_relation(relation: str, triples: List[Triple]) -> Triple:
    """
    Look up a triple in a list of triples by the middle value.
    """
    for triple in triples:
        if triple[1] == relation:
            return triple
    return None


def create_chunk_node_table(connection: kuzu.Connection) -> None:
    # For now, the additional `properties` dict from LlamaIndex is stored as a string
    # TODO: See if it makes sense to add better support for property metadata as columns
    if "Chunk" not in connection._get_node_table_names():
        connection.execute(
            f"""
            CREATE NODE TABLE Chunk (
                id STRING,
                text STRING,
                label STRING,
                embedding DOUBLE[],
                creation_date DATE,
                last_modified_date DATE,
                file_name STRING,
                file_path STRING,
                file_size INT64,
                file_type STRING,
                ref_doc_id STRING,
                PRIMARY KEY(id)
            )
            """
        )


def create_entity_node_tables(connection: kuzu.Connection, entities: List[str]) -> None:
    for tbl_name in entities:
        # For now, the additional `properties` dict from LlamaIndex is stored as a string
        # TODO: See if it makes sense to add better support for property metadata as columns
        if tbl_name not in connection._get_node_table_names():
            connection.execute(
                f"""
                CREATE NODE TABLE {tbl_name} (
                    id STRING,
                    name STRING,
                    label STRING,
                    embedding DOUBLE[],
                    creation_date DATE,
                    last_modified_date DATE,
                    file_name STRING,
                    file_path STRING,
                    file_size INT64,
                    file_type STRING,
                    triplet_source_id STRING,
                    PRIMARY KEY(id)
                )
                """
            )


def create_relation_tables(
    connection: kuzu.Connection, entities: List[str], relationship_schema: List[Triple]
) -> None:
    rel_tables = [tbl["name"] for tbl in connection._get_rel_table_names()]
    # We use Kùzu relationship table group creation DDL commands to create relationship tables
    ddl = ""
    if not any("LINKS" in table for table in rel_tables):
        ddl = "CREATE REL TABLE GROUP LINKS ("
        table_names = []
        for src, _, dst in relationship_schema:
            table_names.append(f"FROM {src} TO {dst}")
        for entity in entities:
            table_names.append(f"FROM Chunk TO {entity}")
        table_names = list(set(table_names))
        ddl += ", ".join(table_names)
        # Add common properties for all the tables here
        ddl += ", label STRING, triplet_source_id STRING)"
    if ddl:
        connection.execute(ddl)

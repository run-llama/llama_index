from pydantic import BaseModel
from typing import Optional, Dict, List


class AlphaMatrix(BaseModel):
    """
    This class is not necessary to understand to use a KodaRetriever - as it will be automatically instantiated if a dictionary is provided.

    Pydantic class to enforce the required fields for a KodaRetriever
    Its best to just instantiate this using a dictionary, don't both trying to instantiate by declaring any AlphaCategory objects.

    Example:
        >>> data = {
                "normal query": { # examples is not required if you aren't using few-shot auto-routing
                    "alpha": .5
                    , "description": "This is a normal query" # desc is not required if you aren't using few-shot auto-routing
                    , "examples": ["This is a normal query", "Another normal query"]
                }
            }
        >>> matrix = AlphaMatrix(data=data) # arg must be named matrix for the retriever to use it

    """

    class AlphaCategory(BaseModel):
        """
        Subclass to enforce the required fields for a category in the AlphaMatrix - necessary for nesting in the AlphaMatrix class
        You should not have to really touch this, as it is only used for type checking and validation.
        """

        alpha: float
        description: Optional[str] = (
            None  # optional if providing a custom LLM, its presumed this was part of your training data for the custom model
        )
        examples: Optional[List[str]] = (
            None  # if not providing a custom model, this is required
        )

    data: Dict[str, AlphaCategory]

    def get_alpha(self, category: str) -> float:
        """Simple helper function to get the alpha value for a given category."""
        if category not in self.data:
            err = f"Provided category '{category}' cannot be found"
            raise ValueError(err)

        return self.data.get(category).alpha  # type: ignore

    def get_examples(self, category: str) -> List[str]:
        """Simple helper function to get the examples for a given category."""
        if category not in self.data:
            err = f"Provided category '{category}' cannot be found"
            raise ValueError(err)

        return self.data.get(category).examples  # type: ignore

    def get_description(self, category: str) -> str:
        """Simple helper function to get the description for a given category."""
        if category not in self.data:
            err = f"Provided category '{category}' cannot be found"
            raise ValueError(err)

        return self.data.get(category).description  # type: ignore

    def get_categories(self) -> list:
        """Simple helper function to get the categories for a given category."""
        return list(self.data.keys())

    def format_category(self, category: str) -> str:
        """Simple helper function to format the category information for a given category."""
        if category not in self.data:
            err = f"Provided category '{category}' cannot be found"
            raise ValueError(err)

        description = self.get_description(category)
        examples = self.get_examples(category)

        category_info = f"""
        - {category}:
            description: {description}
        """.strip()

        if examples:
            examples = "; ".join(examples)
            example_info = f"""
            examples:
                {examples}
            """
            category_info = f"{category_info}\n{example_info}"

        return category_info

    def get_all_category_info(self) -> str:
        """Simple helper function to get the category information for all categories."""
        categories = []
        for category in self.get_categories():
            category_info = self.format_category(category)
            categories.append(category_info)
        return "\n".join(categories)

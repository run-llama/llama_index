"""
Chain of table.

All prompts adapted from original paper by Wang et al.:
https://arxiv.org/pdf/2401.04398v1.pdf

"""

import re
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.base.query_pipeline.query import QueryComponent
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.query_pipeline.query import QueryPipeline as QP
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI


def _get_regex_parser_fn(regex: str) -> Callable:
    """Get regex parser."""

    def _regex_parser(output: Any) -> List[str]:
        """Regex parser."""
        output = str(output)
        m = re.search(regex, output)
        args = m.group(1)
        if "," in args:
            return [a.strip().strip("'\"") for a in args.split(",")]
        else:
            return [args.strip().strip("'\"")]

    return _regex_parser


class FunctionSchema(BaseModel):
    """Function schema."""

    prompt: PromptTemplate = Field(..., description="Prompt.")
    regex: Optional[str] = Field(default=None, description="Regex.")

    @abstractmethod
    def fn(self, table: pd.DataFrame, args: Any) -> Callable:
        """Function."""
        raise NotImplementedError

    def parse_args(self, args: str) -> Any:
        """Parse args."""
        regex_fn = _get_regex_parser_fn(self.regex)
        return regex_fn(args)

    def parse_args_and_call_fn(self, table: pd.DataFrame, args: str) -> pd.DataFrame:
        """Parse args and call function."""
        args = self.parse_args(args)
        return args, self.fn(table, args)

    def generate_prompt_component(self, **kwargs: Any) -> QueryComponent:
        """Generate prompt."""
        # add valid kwargs to prompt
        new_kwargs = {}
        for key in kwargs:
            if key in self.prompt.template_vars:
                new_kwargs[key] = kwargs[key]
        return self.prompt.as_query_component(partial=new_kwargs)


dynamic_plan_str = """\
========================================= Atomic Operations =========================================
If the table needs an extra inferred column to answer the question, we use f_add_column() to
add this column. For example,
/*
col : Week | When | Kickoff | Opponent | Results; Final score | Results; Team record
row 1 : 1 | Saturday, April 13 | 7:00 p.m. | at Rhein Fire | W 27-21 | 1-0
row 2 : 2 | Saturday, April 20 | 7:00 p.m. | London Monarchs | W 37-3 | 2-0
row 3 : 3 | Sunday, April 28 | 6:00 p.m. | at Barcelona Dragons | W 33-29 | 3-0
*/
Question : what is the date of the competition with highest attendance?
The existing columns are: "Week", "When", "Kickoff", "Opponent", "Results; Final score",
"Results; Team record", "Game site", "Attendance".
Function : f_add_column(Attendance number)
Explanation: the question asks about the date of the competition with highest score. Each
row is about one competition. We extract the value from column "Attendance" and create a
different column "Attendance number" for each row. The datatype is Numerical.

If the table only needs a few rows to answer the question, we use f_select_row() to select
these rows for it. For example,
/*
col : Home team | Home Team Score | Away Team | Away Team Score | Venue | Crowd
row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836
row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154
row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651
*/
Question : Whose home team score is higher, richmond or st kilda?
Function : f_select_row(row 1, row 3)
Explanation: The question asks about the home team score of richmond and st kilda. We need
to know the information of richmond and st kilda in row 1 and row 3. We select row 1
and row 3.

If the table only needs a few columns to answer the question, we use
f_select_column() to select these columns for it. For example,
/*
col : Competition | Total Matches | Cardiff Win | Draw | Swansea Win
row 1 : League | 55 | 19 | 16 | 20
row 2 : FA Cup | 2 | 0 | 27 | 2
row 3 : League Cup | 5 | 2 | 0 | 3
*/
Question : Are there cardiff wins that have a draw greater than 27?
Function : f_select_column([cardiff win, draw])
Explanation: The question asks about the cardiff wins that have a draw greater than 27.
    We need to know the information of cardiff win and draw. We select column cardiff win and
    draw.

If the question asks about items with the same value and the number of these items, we use
f_group_by() to group the items. For example,
/*
col : Rank | Lane | Athlete | Time | Country
row 1 : 1 | 6 | Manjeet Kaur (IND) | 52.17 | IND
row 2 : 2 | 5 | Olga Tereshkova (KAZ) | 51.86 | KAZ
row 3 : 3 | 4 | Pinki Pramanik (IND) | 53.06 | IND
*/
Question: tell me the number of athletes from japan.
Function : f_group_by(Country)
Explanation: The question asks about the number of athletes from India. Each row is about
an athlete. We can group column "Country" to group the athletes from the same country.

If the question asks about the order of items in a column, we use f_sort_by() to sort
the items. For example,
/*
col : Position | Club | Played | Points | Wins | Draws | Losses | Goals for | Goals against
row 1 : 1 | Malaga CF | 42 | 79 | 22 | 13 | 7 | 72 | 47
row 10 : 10 | CP Merida | 42 | 59 | 15 | 14 | 13 | 48 | 41
row 3 : 3 | CD Numancia | 42 | 73 | 21 | 10 | 11 | 68 | 40
*/
Question: what club placed in the last position?
Function : f_sort_by(Position)
Explanation: the question asks about the club in the last position. Each row is about a
club. We need to know the order of position from last to front. There is a column for
position and the column name is Position. The datatype is Numerical.

========================================= Operation Chain Task+Examples =========================================

Your task is to construct an operation chain using the above operations to answer the questions.

Some rules:
- The operation chain must end with <END>.
- Please use arrow -> to separate operations.
- You can use any operation any number of times, in any order.
- If the operation chain is incomplete, you must help complete it by adding the missing \
    operation. For example in the below example, if the operation chain is \
    'f_add_column(Date) -> f_select_row([row 1, row 2]) -> f_select_column([Date, League]) -> ' \
    then you must add the following: 'f_sort_by(Date) -> <END>'
- If the table is simplified/reduced enough to answer the question, ONLY WRITE <END>. \
For instance, if the table is only 1 row or a small set of columns, PLEASE write \
<END> - DON'T DO unnecessary operations.

Here are some examples.
/*
col : Date | Division | League | Regular Season | Playoffs | Open Cup
row 1 : 2001/01/02 | 2 | USL A-League | 4th, Western | Quarterfinals | Did not qualify
row 2 : 2002/08/06 | 2 | USL A-League | 2nd, Pacific | 1st Round | Did not qualify
row 5 : 2005/03/24 | 2 | USL First Division | 5th | Quarterfinals | 4th Round
*/
Question: what was the last year where this team was a part of the usl a-league?
Candidates: {candidates}
Previous Function Chain: f_add_column(Date) -> f_select_row([row 1, row 2, row 5])
Function Chain: f_select_column([Date, League]) -> f_sort_by(Date) -> <END>

/*
col : Rank | Cyclist | Country
row 3 : 3 | Davide Rebellin (ITA) | ITA
row 4 : 4 | Paolo Bettini (ITA) | ITA
*/
Question: Which italian cyclist placed in the top 10?
Candidates: {candidates}
Previous Function Chain: f_add_column(Country) -> f_select_row([row 3, row 4]) -> f_select_column([Rank, Cyclist, Country])
Function Chain: <END>

/*
{serialized_table}
*/
Question: {question}
Candidates: {candidates}
Previous Function Chain: {incomplete_function_chain}
Function Chain: """

dynamic_plan_prompt = PromptTemplate(dynamic_plan_str)


## function prompts
add_column_str = """\
To answer the question, we can first use f_add_column() to add more columns to the table.
The added columns should have these data types:
1. Numerical: the numerical strings that can be used in sort, sum
2. Datetype: the strings that describe a date, such as year, month, day
3. String: other strings
/*
col : Week | When | Kickoff | Opponent | Results; Final score | Results; Team record
row 1 : 1 | Saturday, April 13 | 7:00 p.m. | at Rhein Fire | W 27-21 | 1-0
row 2 : 2 | Saturday, April 20 | 7:00 p.m. | London Monarchs | W 37-3 | 2-0
row 3 : 3 | Sunday, April 28 | 6:00 p.m. | at Barcelona Dragons | W 33-29 | 3-0
*/
Question: what is the date of the competition with highest attendance?
The existing columns are: "Week", "When", "Kickoff", "Opponent", "Results; Final score",
"Results; Team record", "Game site", "Attendance".
Explanation: the question asks about the date of the competition with highest score. Each
row is about one competition. We extract the value from column "Attendance" and create a
different column "Attendance number" for each row. The datatype is Numerical.
Therefore, the answer is: f_add_column(Attendance number). The value: 32092 | 34186 | 17503
/*
col : Rank | Lane | Player | Time
row 1 : 5 | Olga Tereshkova (KAZ) | 51.86
row 2 : 6 | Manjeet Kaur (IND) | 52.17
row 3 : 3 | Asami Tanno (JPN) | 53.04
*/
Question: tell me the number of athletes from japan.
The existing columns are: Rank, Lane, Player, Time.
Explanation: the question asks about the number of athletes from japan. Each row is about
one athlete. We need to know the country of each athlete. We extract the value from column
"Player" and create a different column "Country of athletes" for each row. The datatype
is String.
Therefore, the answer is: f_add_column(Country of athletes). The value: KAZ | IND | JPN
{serialized_table}
Question: {question}
Explanation: """


class AddColumnSchema(FunctionSchema):
    """Add column schema."""

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        prompt = PromptTemplate(add_column_str)
        regex = "f_add_column\\((.*)\\)"
        super().__init__(
            prompt=prompt,
            regex=regex,
            **kwargs,
        )

    def fn(self, table: pd.DataFrame, args: Any) -> pd.DataFrame:
        """Call function."""
        col_name = args["col_name"]
        col_values = args["col_values"]
        table = table.copy()
        # add column to table with col_name and col_values
        table[col_name] = col_values
        return table

    def parse_args(self, args: str) -> Any:
        """Parse args."""
        regex_fn = _get_regex_parser_fn(self.regex)
        args = regex_fn(args)

        value_args_regex = "value:(.*)"
        value_regex_fn = _get_regex_parser_fn(value_args_regex)
        value_args = value_regex_fn(args)

        return {
            "col_name": args,
            "col_values": value_args,
        }

    def parse_args_and_call_fn(self, table: pd.DataFrame, args: str) -> pd.DataFrame:
        """Parse args and call function."""
        args = self.parse_args(args)
        return [args["col_name"]], self.fn(table, args)


add_column_schema = AddColumnSchema()


select_column_str = """\
Use f_select_column() to filter out useless columns in the table according to information
in the statement and the table.

Additional rules:
- You must ONLY select from the valid set of columns, in the first row of the table marked with "col : ...".
- You must NOT select the same column multiple times.
- You must NOT select a row (e.g. select_column(League) in the example below is not allowed)

/*
col : competition | total matches | cardiff win | draw | swansea win
row 1 : League | 55 | 19 | 16 | 20
row 2 : FA Cup | 2 | 0 | 27 | 2
row 3 : League Cup | 5 | 2 | 0 | 3
*/
Question : Are there cardiff wins that have a draw greater than 27?
similar words link to columns :
no cardiff wins -> cardiff win
a draw -> draw
column value link to columns :
27 -> draw
semantic sentence link to columns :
None
The answer is : f_select_column([cardiff win, draw])

/*
{serialized_table}
*/
Question : {question}
"""


class SelectColumnSchema(FunctionSchema):
    """Select column schema."""

    def __init__(self, **kwargs: Any) -> None:
        """Init params."""
        prompt = PromptTemplate(select_column_str)
        super().__init__(
            prompt=prompt,
            regex="f_select_column\\(\\[(.*)\\]\\)",
            **kwargs,
        )

    def fn(self, table: pd.DataFrame, args: Any) -> pd.DataFrame:
        """Call function."""
        # assert that args is a list
        assert isinstance(args, list)
        table = table.copy()
        # select columns from table
        return table[args]


select_column_schema = SelectColumnSchema()


# select_args_str = """\
# Using f_select_row() to select relevant rows in the given table that support or oppose the
# statement.
# Please use f_select_row([*]) to select all rows in the table.
# /*
# table caption : 1972 vfl season.
# col : home team | home team score | away team | away team score | venue | crowd
# row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836
# row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154
# row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651
# row 4 : geelong | 17.10 (112) | collingwood | 17.9 (111) | kardinia park | 23108
# row 5 : north melbourne | 8.12 (60) | carlton | 23.11 (149) | arden street oval | 11271
# row 6 : hawthorn | 15.16 (106) | essendon | 12.15 (87) | vfl park | 36749
# */
# statement : what is the away team with the highest score?
# explain : the statement want to ask the away team of highest away team score. the highest
# away team score is 23.11 (149). it is on the row 5.so we need row 5.
# The answer is : f_select_row([row 5])
# """
# select_args_prompt = PromptTemplate(select_args_str)
# select_args_schema = FunctionSchema(
#     prompt=select_args_str,
#     regex="f_select_row\([(.*)]\)",
# )

select_row_str = """\
Using f_select_row() to select relevant rows in the given table that support or oppose the
statement.
Please use f_select_row([*]) to select all rows in the table.
/*
table caption : 1972 vfl season.
col : home team | home team score | away team | away team score | venue | crowd
row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836
row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154
row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651
row 4 : geelong | 17.10 (112) | collingwood | 17.9 (111) | kardinia park | 23108
row 5 : north melbourne | 8.12 (60) | carlton | 23.11 (149) | arden street oval | 11271
row 6 : hawthorn | 15.16 (106) | essendon | 12.15 (87) | vfl park | 36749
*/
statement : what is the away team with the highest score?
explain : the statement want to ask the away team of highest away team score. the highest
away team score is 23.11 (149). it is on the row 5.so we need row 5.
The answer is : f_select_row([row 5])

{serialized_table}
statement : {question}
explain : \
"""


class SelectRowSchema(FunctionSchema):
    """Select row schema."""

    def __init__(self, **kwargs: Any) -> None:
        """Init params."""
        prompt = PromptTemplate(select_row_str)
        super().__init__(
            prompt=prompt,
            regex="f_select_row\\(\\[(.*)\\]\\)",
            **kwargs,
        )

    def fn(self, table: pd.DataFrame, args: Any) -> pd.DataFrame:
        """Call function."""
        # assert that args is a list
        assert isinstance(args, list)
        # parse out args since it's in the format ["row 1", "row 2"], etc.
        args = [int(arg.split(" ")[1]) - 1 for arg in args]

        table = table.copy()
        # select rows from table
        return table.loc[args]


select_row_schema = SelectRowSchema()


group_by_str = """\
To answer the question, we can first use f_group_by() to group the values in a column.
/*
col : Rank | Lane | Athlete | Time | Country
row 1 : 1 | 6 | Manjeet Kaur (IND) | 52.17 | IND
row 2 : 2 | 5 | Olga Tereshkova (KAZ) | 51.86 | KAZ
row 3 : 3 | 4 | Pinki Pramanik (IND) | 53.06 | IND
row 4 : 4 | 1 | Tang Xiaoyin (CHN) | 53.66 | CHN
row 5 : 5 | 8 | Marina Maslyonko (KAZ) | 53.99 | KAZ
*/
Question: tell me the number of athletes from japan.
The existing columns are: Rank, Lane, Athlete, Time, Country.
Explanation: The question asks about the number of athletes from India. Each row is about
an athlete. We can group column "Country" to group the athletes from the same country.
Therefore, the answer is: f_group_by(Country).

{serialized_table}
Question: {question}
Explanation: """


class GroupBySchema(FunctionSchema):
    """Group by fn schema."""

    def __init__(self, **kwargs: Any) -> None:
        """Init params."""
        prompt = PromptTemplate(group_by_str)
        super().__init__(
            prompt=prompt,
            regex="f_group_by\\((.*)\\)",
            **kwargs,
        )

    def fn(self, table: pd.DataFrame, args: Any) -> pd.DataFrame:
        """Call function."""
        # assert that args is a string
        assert isinstance(args, list) and len(args) == 1
        args = str(args[0])

        table = table.copy()
        # group by column
        return table.groupby(args).count()


group_by_schema = GroupBySchema()


sort_by_str = """\
To answer the question, we can first use f_sort_by() to sort the values in a column to get
the
order of the items. The order can be "large to small" or "small to large".
The column to sort should have these data types:
1. Numerical: the numerical strings that can be used in sort
2. DateType: the strings that describe a date, such as year, month, day
3. String: other strings
/*
col : Position | Club | Played | Points | Wins | Draws | Losses | Goals for | Goals against
row 1 : 1 | Malaga CF | 42 | 79 | 22 | 13 | 7 | 72 | 47
row 10 : 10 | CP Merida | 42 | 59 | 15 | 14 | 13 | 48 | 41
row 3 : 3 | CD Numancia | 42 | 73 | 21 | 10 | 11 | 68 | 40
*/

More rules:
- The answer MUST be in the format "the answer is: f_sort_by(Arg1)", where Arg1 is the
column name.
- The answer CANNOT include multiple columns
- You CANNOT run f_sort_by on a row. For instance, f_sort_by(row 1) is not allowed.

Question: what club placed in the last position?
The existing columns are: Position, Club, Played, Points, Wins, Draws, Losses, Goals for,
Goals against
Explanation: the question asks about the club in the last position. Each row is about a
club. We need to know the order of position from last to front. There is a column for
position and the column name is Position. The datatype is Numerical.
Therefore, the answer is: f_sort_by(Position), the order is "large to small".

{serialized_table}
Question: {question}
Explanation: """


class SortBySchema(FunctionSchema):
    """Sort by fn schema."""

    def __init__(self, **kwargs: Any) -> None:
        """Init params."""
        prompt = PromptTemplate(sort_by_str)
        super().__init__(
            prompt=prompt,
            regex="f_sort_by\\((.*)\\)",
            **kwargs,
        )

    def fn(self, table: pd.DataFrame, args: Any) -> pd.DataFrame:
        """Call function."""
        # assert that args is a string
        assert isinstance(args, list) and len(args) == 1
        args = str(args[0])

        table = table.copy()
        # sort by column
        return table.sort_values(args)


sort_by_schema = SortBySchema()


query_prompt_str = """\
========================================= Prompt =========================================
Here is the table to answer this question. Please understand the table and answer the
question:
/*
col : Rank | City | Passengers Number | Ranking | Airline
row 1 : 1 | United States, Los Angeles | 14749 | 2 | Alaska Airlines
row 2 : 2 | United States, Houston | 5465 | 8 | United Express
row 3 : 3 | Canada, Calgary | 3761 | 5 | Air Transat, WestJet
row 4 : 4 | Canada, Saskatoon | 2282 | 4 |
row 5 : 5 | Canada, Vancouver | 2103 | 2 | Air Transat
row 6 : 6 | United States, Phoenix | 1829 | 1 | US Airways
row 7 : 7 | Canada, Toronto | 1202 | 1 | Air Transat, CanJet
row 8 : 8 | Canada, Edmonton | 110 | 2 |
row 9 : 9 | United States, Oakland | 107 | 5 |
*/
Question: how many more passengers flew to los angeles than to saskatoon from manzanillo
airport in 2013?
The answer is: 12467

Here is the table to answer this question. Please understand the table and answer the
question:
/*
Group ID | Country | Count
1 | ITA | 3
2 | ESP | 3
3 | RUS | 2
4 | FRA | 2
*/
Question: which country had the most cyclists in top 10?
The answer is: Italy.

Here is the table to answer this question. Please understand the table and answer the
question:
{serialized_table}
Question: {question}
The answer is: """
query_prompt = PromptTemplate(query_prompt_str)


schema_mappings: Dict[str, FunctionSchema] = {
    "f_add_column": add_column_schema,
    "f_select_column": select_column_schema,
    "f_select_row": select_row_schema,
    "f_group_by": group_by_schema,
    "f_sort_by": sort_by_schema,
}


def _dynamic_plan_parser(dynamic_plan: Any) -> Dict[str, Any]:
    """Parse dynamic plan."""
    dynamic_plan_str = str(dynamic_plan)
    # break out arrows
    tokens = dynamic_plan_str.split("->")
    # look at first token
    first_token = tokens[0].strip().lower()
    for key in schema_mappings:
        if key in first_token:
            return key
    # look at end token
    if "<END>" in tokens[0]:
        return "<END>"
    raise ValueError(f"Could not parse dynamic plan: {dynamic_plan_str}")


def serialize_chain(op_chain: List[Tuple[str, str]]) -> str:
    """
    Serialize operation chain.

    Operation chain is list of (fn, args) tuples.

    Return string in form: fn1(args1) -> fn2(args2) -> ...

    Leave dangling arrow at end.

    """
    # implement
    output_str = ""
    for op in op_chain:
        output_str += f"{op[0]}({op[1]}) -> "
    return output_str


def serialize_keys(keys: Any) -> str:
    """Serialize keys."""
    return ", ".join(list(keys))


def serialize_table(table: pd.DataFrame) -> str:
    """Serialize table."""
    # return table.to_markdown(tablefmt="github")

    def _esc_newl(s: str) -> str:
        """Escape newlines."""
        return s.replace("\n", "\\n")

    output_str = f"col : {' | '.join([_esc_newl(c) for c in table.columns])}\n"
    for i in range(len(table)):
        output_str += (
            f"row {i + 1} : {' | '.join([_esc_newl(str(x)) for x in table.iloc[i]])}\n"
        )
    return output_str


class ChainOfTableQueryEngine(CustomQueryEngine):
    """Chain of table query engine."""

    dynamic_plan_prompt: PromptTemplate = Field(
        default=dynamic_plan_prompt, description="Dynamic plan prompt."
    )
    query_prompt: PromptTemplate = Field(
        default=query_prompt, description="Query prompt."
    )
    table: pd.DataFrame = Field(..., description="Table (in pandas).")
    llm: LLM = Field(..., description="LLM")
    max_iterations: int = Field(default=10, description="Max iterations.")
    verbose: bool = Field(default=False, description="Verbose.")

    def __init__(
        self,
        table: pd.DataFrame,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        llm = llm or OpenAI(model="gpt-3.5-turbo")
        super().__init__(table=table, llm=llm, verbose=verbose, **kwargs)

    def custom_query(self, query_str: str) -> Response:
        """Run chain of table query engine."""
        op_chain = []
        dynamic_plan_parser = FnComponent(fn=_dynamic_plan_parser)

        cur_table = self.table.copy()

        for iter in range(self.max_iterations):
            if self.verbose:
                print_text(f"> Iteration: {iter}\n", color="green")
                print_text(
                    f"> Current table:\n{serialize_table(cur_table)}\n\n", color="blue"
                )
            # generate dynamic plan
            dynamic_plan_prompt = self.dynamic_plan_prompt.as_query_component(
                partial={
                    "serialized_table": serialize_table(cur_table),
                    "candidates": serialize_keys(schema_mappings.keys()),
                    "incomplete_function_chain": serialize_chain(op_chain),
                }
            )

            dynamic_plan_chain = QP(
                chain=[dynamic_plan_prompt, self.llm, dynamic_plan_parser],
                callback_manager=self.callback_manager,
            )
            key = dynamic_plan_chain.run(question=query_str)
            if key == "<END>":
                if self.verbose:
                    print("> Ending operation chain.")
                break

            # generate args from key
            fn_prompt = schema_mappings[key].generate_prompt_component(
                serialized_table=serialize_table(cur_table),
            )
            generate_args_chain = QP(
                chain=[fn_prompt, self.llm], callback_manager=self.callback_manager
            )
            raw_args = generate_args_chain.run(question=query_str)
            args, cur_table = schema_mappings[key].parse_args_and_call_fn(
                cur_table, raw_args
            )

            op_chain.append((key, args))
            if self.verbose:
                print_text(f"> New Operation + Args: {key}({args})\n", color="pink")
                print_text(
                    f"> Current chain: {serialize_chain(op_chain)}\n", color="pink"
                )

        # generate query prompt
        query_prompt = self.query_prompt.as_query_component(
            partial={
                "serialized_table": serialize_table(cur_table),
            }
        )
        query_chain = QP(
            chain=[query_prompt, self.llm], callback_manager=self.callback_manager
        )
        response = query_chain.run(question=query_str)
        return Response(response=str(response))


class ChainOfTablePack(BaseLlamaPack):
    """Chain of table pack."""

    def __init__(
        self,
        table: pd.DataFrame,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.query_engine = ChainOfTableQueryEngine(
            table=table,
            llm=llm,
            verbose=verbose,
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "query_engine": self.query_engine,
            "llm": self.query_engine.llm,
            "query_prompt": self.query_engine.query_prompt,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)

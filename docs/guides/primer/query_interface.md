
# Query Interface
Querying an index or a graph involves a three main components:

- **Retrievers**: A retriever class retrieves a set of Nodes from an index given a query.
- **Response Synthesizer**: This class takes in a set of Nodes and synthesizes an answer given a query.
- **Query Engine**: This class takes in a query and returns a Response object. It can make use
   of Retrievers and Response Synthesizer modules under the hood.

![](/_static/query/query_classes.png)


## Design Philosophy: Progressive Disclosure of Complexity

Progressive disclosure of complexity is a design philosophy that aims to strike 
a balance between the needs of beginners and experts. The idea is that you should 
give users the simplest and most straightforward interface or experience possible 
when they first encounter a system or product, but then gradually reveal more 
complexity and advanced features as users become more familiar with the system. 
This can help prevent users from feeling overwhelmed or intimidated by a system 
that seems too complex, while still giving experienced users the tools they need 
to accomplish advanced tasks.

![](/_static/query/disclosure.png)


In the case of LlamaIndex, we've tried to balance simplicity and complexity by 
providing a high-level API that's easy to use out of the box, but also a low-level 
composition API that gives experienced users the control they need to customize the 
system to their needs. By doing this, we hope to make LlamaIndex accessible to 
beginners while still providing the flexibility and power that experienced users need.

## Resources

- The basic query interface over an index is found in our [usage pattern guide](/guides/primer/usage_pattern.md). The guide
  details how to specify parameters for a retriever/synthesizer/query engine over a 
  single index structure.
- A more advanced query interface is found in our [composability guide](/how_to/index/composability.md). The guide
  describes how to specify a graph over multiple index structures.
- We also provide a guide to some of our more [advanced components](/how_to/query_engine/advanced/root.md), which can be added 
  to a retriever or a query engine. See our [query transformations](/how_to/query_engine/advanced/query_transformations.md)
  and 
  [second-stage processing](/how_to/query_engine/advanced/second_stage.md) modules. 


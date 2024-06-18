# Building an LLM application

Welcome to the beginning of Understanding LlamaIndex. This is a series of short, bite-sized tutorials on every stage of building an LLM application to get you acquainted with how to use LlamaIndex before diving into more advanced and subtle strategies. If you're an experienced programmer new to LlamaIndex, this is the place to start.

## Key steps in building an LLM application

!!! tip
    If you've already read our [high-level concepts](../getting_started/concepts.md) page you'll recognize several of these steps.

There are a series of key steps involved in building any LLM-powered application, whether it's answering questions about your data, creating a chatbot, or an autonomous agent. Throughout our documentation, you'll notice sections are arranged roughly in the order you'll perform these steps while building your app. You'll learn about:

- **[Using LLMs](./using_llms/using_llms.md)**: whether it's OpenAI or any number of hosted LLMs or a locally-run model of your own, LLMs are used at every step of the way, from indexing and storing to querying and parsing your data. LlamaIndex comes with a huge number of reliable, tested prompts and we'll also show you how to customize your own.

- **[Loading](./loading/loading.md)**: getting your data from wherever it lives, whether that's unstructured text, PDFs, databases, or APIs to other applications. LlamaIndex has hundreds of connectors to every data source over at [LlamaHub](https://llamahub.ai/).

- **[Indexing](./indexing/indexing.md)**: once you've got your data there are an infinite number of ways to structure access to that data to ensure your applications is always working with the most relevant data. LlamaIndex has a huge number of these strategies built-in and can help you select the best ones.

- **[Storing](./storing/storing.md)**: you will probably find it more efficient to store your data in indexed form, or pre-processed summaries provided by an LLM, often in a specialized database known as a `Vector Store` (see below). You can also store your indexes, metadata and more.

- **[Querying](./querying/querying.md)**: every indexing strategy has a corresponding querying strategy and there are lots of ways to improve the relevance, speed and accuracy of what you retrieve and what the LLM does with it before returning it to you, including turning it into structured responses such as an API.

- **[Putting it all together](./putting_it_all_together/index.md)**: whether you are building question & answering, chatbots, an API, or an autonomous agent, we show you how to get your application into production.

- **[Tracing and debugging](./tracing_and_debugging/tracing_and_debugging.md)**: also called **observability**, it's especially important with LLM applications to be able to look into the inner workings of what's going on to help you debug problems and spot places to improve.

- **[Evaluating](./evaluating/evaluating.md)**: every strategy has pros and cons and a key part of building, shipping and evolving your application is evaluating whether your change has improved your application in terms of accuracy, performance, clarity, cost and more. Reliably evaluating your changes is a crucial part of LLM application development.

## Let's get started!

Ready to dive in? Head to [using LLMs](./using_llms/using_llms.md).

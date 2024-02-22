var exampleData = `<div class="container">
  <h3 class="example-heading toggle-example">Usage Example</h3>
  <div class="usage-examples hidden">
      <div class="example">
        \`\`\`python
        from llama_index.core import Settings

        # global default
        Settings.llm = llm

        # per-component
        # objects that use an LLM accept it as a kwarg
        index.as_query_engine(llm=llm)

        index.as_chat_engine(llm=llm)
        \`\`\`
      </div>
  </div>
</div>`;

var exampleElement = document.createElement("div");
exampleElement.innerHTML = exampleData;

function addToggleToExample() {
  const toggleExample = document.querySelector(".toggle-example");
  const usageExamples = document.querySelector(".usage-examples");

  toggleExample.addEventListener("click", function () {
    console.log("clicked!");
    console.log(usageExamples);
    usageExamples.classList.toggle("hidden");
  });
}

document$.subscribe(function () {
  console.log("document loaded");
  console.log(window.location.pathname);
  if (window.location.pathname.includes("/LLMS/")) {
    document.querySelector(".md-content__inner").prepend(exampleElement);
    addToggleToExample();
  }
});

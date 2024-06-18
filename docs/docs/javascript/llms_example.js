var exampleTemplate = `<h3 class="example-heading toggle-example">Framework Usage</h3>
  <div class="usage-examples hidden">
      <div class="example">
      </div>
  </div>`;

var exampleMarkdown = `\`\`\`python
from llama_index.core import Settings

# global default
Settings.llm = llm

# per-component
# objects that use an LLM accept it as a kwarg
index.as_query_engine(llm=llm)

index.as_chat_engine(llm=llm)
\`\`\``;

function addToggleToExample() {
  const toggleExample = document.querySelector(".toggle-example");
  const usageExamples = document.querySelector(".usage-examples");

  toggleExample.addEventListener("click", function () {
    console.log("clicked!");
    console.log(usageExamples);
    usageExamples.classList.toggle("hidden");
  });
}

// Add marked package as <script> tag
var script = document.createElement("script");
script.type = "text/javascript";
script.async = true;
script.onload = function () {
  document$.subscribe(function () {
    console.log("document loaded");
    console.log(window.location.pathname);

    if (window.location.pathname.includes("/LLMS/")) {
      var exampleElement = document.createElement("div");
      exampleElement.className = "container";
      exampleElement.innerHTML = exampleTemplate;
      exampleElement.children[1].children[0].innerHTML =
        marked.parse(exampleMarkdown);
      document.querySelector(".md-content__inner").prepend(exampleElement);
      addToggleToExample();
    }
  });
};
script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
document.head.appendChild(script);

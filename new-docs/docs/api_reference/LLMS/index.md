# LLM

<div class="container">
    <h3 class="example-heading toggle-example">Usage Example</h3>
    <div class="usage-examples hidden">
        <div class="example">
          ```python
          from llama_index.core import Settings

          # global default
          Settings.llm = llm

          # per-component
          # objects that use an LLM accept it as a kwarg
          index.as_query_engine(llm=llm)

          index.as_chat_engine(llm=llm)
          ```
        </div>
    </div>

</div>

<style>
/* CSS styles for side-by-side layout */
.container {
    display: flex-col;
    justify-content: space-between;
    margin-bottom: 20px; /* Adjust spacing between sections */
    position: sticky;
    top: 2.4rem;
    z-index: 1000; /* Ensure it's above other content */
    background-color: white; /* Match your page background */
    padding: 0.2rem;
}

.example-heading {
  margin: 0.2rem !important;
}

.usage-examples {
    width: 100%; /* Adjust the width as needed */
    border: 1px solid var(--md-default-fg-color--light);
    border-radius: 2px;
    padding: 0.2rem;
}

/* Additional styling for the toggle */
.toggle-example {
    cursor: pointer;
    color: white;
    text-decoration: underline;
    background-color: var(--md-primary-fg-color);
    padding: 0.2rem;
    border-radius: 2px;
}

.hidden {
  display: none;
}

</style>

<script>
// JavaScript for toggling the usage example section
document.addEventListener('DOMContentLoaded', function () {
    const toggleExample = document.querySelector('.toggle-example');
    const usageExamples = document.querySelector('.usage-examples');

    toggleExample.addEventListener('click', function () {
        console.log('clicked!');
        console.log(usageExamples)
        usageExamples.classList.toggle('hidden');
    });
});
</script>

::: llama_index.core.llms.llm
options:
inherited_members: true
members: - LLM
show_source: false

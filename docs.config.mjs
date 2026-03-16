export default {
  section: "llama-index",
  label: "LlamaIndex Framework",
  content: [
    { src: "./docs/src/content/docs/framework", dest: "python/framework" },
  ],
  sidebar: [
    {
      label: "LlamaIndex Framework",
      content: {
        type: "autogenerate",
        directory: "python/framework",
        collapsed: true,
      },
    },
  ],
  // Optional: link to API reference
  apiReferenceLink: "/python/framework-api-reference/",
};

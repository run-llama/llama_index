// @ts-check
import { defineConfig, passthroughImageService } from "astro/config";
import starlight from "@astrojs/starlight";
import react from "@astrojs/react";
import AutoImport from "astro-auto-import";
import starlightAutoSidebar from "starlight-auto-sidebar";
import path from "path";
import fs from "fs";
import { visit } from "unist-util-visit";

/** Checks and logs broken links in Markdown files. */
function findBrokenLinks() {
  return /** @param {any} tree @param {any} file */ (tree, file) => {
    const filePath = file && file.path ? String(file.path) : "";
    const currentDir = filePath ? path.dirname(filePath) : "";
    const docsRoot = path.resolve("src/content/docs");

    visit(tree, "link", (node) => {
      const url = typeof node.url === "string" ? node.url : "";
      if (!url) return;

      // ignore anchors
      if (url.startsWith("#")) return;
      // ignore mailto links
      if (/^mailto:/i.test(url)) return;
      // ignore localhost links (http(s), protocol-relative, or bare hostname)
      if (
        /^(?:https?:)?\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?/i.test(
          url,
        )
      )
        return;
      if (/^localhost(?::\d+)?(?:\/?|$)/i.test(url)) return;
      // ignore other external links
      if (/^(?:[a-z]+:)?\/\//i.test(url)) return;

      // strip query/hash for path resolution
      const hashIndex = url.indexOf("#");
      const queryIndex = url.indexOf("?");
      let endIndex = url.length;
      if (hashIndex !== -1) endIndex = Math.min(endIndex, hashIndex);
      if (queryIndex !== -1) endIndex = Math.min(endIndex, queryIndex);
      const rawPath = url.slice(0, endIndex);

      // normalize path and build candidates
      const normalized = rawPath.replace(/\\/g, "/");
      const isAbsolute = normalized.startsWith("/");
      const withoutTrailing = normalized.replace(/\/+$/, "");
      const ext = path.extname(withoutTrailing);

      /**
       * @param {string=} p
       * @returns {string}
       */
      function resolveFrom(p) {
        if (isAbsolute) {
          let relFromRoot = withoutTrailing.replace(/^\//, "");
          // Map site base "/python/" to docs root for existence checks
          const lower = relFromRoot.toLowerCase();
          if (lower.startsWith("python/")) {
            relFromRoot = relFromRoot.slice(7);
          }
          return path.resolve(docsRoot, relFromRoot, p || "");
        }
        return path.resolve(currentDir, withoutTrailing, p || "");
      }

      /** @type {string[]} */
      let candidates = [];

      if (/\.(md|mdx)$/i.test(withoutTrailing)) {
        // explicit markdown link; check directly
        candidates = [resolveFrom("")];
      } else if (ext) {
        // has some other extension; treat as asset – check existence directly
        candidates = [resolveFrom("")];
      } else {
        // pretty URL: try file.md, file.mdx, file/index.md, file/index.mdx
        candidates = [
          resolveFrom("") + ".ipynb",
          resolveFrom("") + ".md",
          resolveFrom("") + ".mdx",
          resolveFrom("index.md"),
          resolveFrom("index.mdx"),
        ];
      }

      const exists = candidates.some((p) => fs.existsSync(p));
      const logUrl = url;
      if (!exists) {
        console.log(`Broken link: ${logUrl}\n  in ${filePath}`);
      } else {
        // console.log(`Valid markdown link: ${logUrl}\n  in ${filePath}`);
      }
    });
  };
}

// https://astro.build/config
export default defineConfig({
  site: "https://developers.llamaindex.ai",
  base: "/python/",
  outDir: path.resolve("dist"),
  integrations: [
    starlight({
      plugins: [starlightAutoSidebar()],
      title: "LlamaIndex Python Documentation",
      head: [
        {
          tag: "script",
          content: `
					(function (w, d, s, l, i) {
					  w[l] = w[l] || [];
					  w[l].push({ "gtm.start": new Date().getTime(), event: "gtm.js" });
					  var f = d.getElementsByTagName(s)[0],
						j = d.createElement(s),
						dl = l != "dataLayer" ? "&l=" + l : "";
					  j.async = true;
					  j.src = "https://www.googletagmanager.com/gtm.js?id=" + i + dl;
					  f.parentNode.insertBefore(j, f);
					})(window, document, "script", "dataLayer", "GTM-WWRFB36R");
				  `,
        },
        {
          tag: "script",
          content: `
						document.addEventListener("DOMContentLoaded", function () {
							var script = document.createElement("script");
							script.type = "module";
							script.id = "runllm-widget-script"
							script.src = "https://widget.runllm.com";
							script.setAttribute("version", "stable");
							script.setAttribute("crossorigin", "true");
							script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
							script.setAttribute("runllm-name", "LlamaIndex");
							script.setAttribute("runllm-position", "BOTTOM_RIGHT");
							script.setAttribute("runllm-assistant-id", "209");
							script.setAttribute("runllm-disable-ask-a-person", true);
							script.setAttribute(
								"runllm-slack-community-url",
								"https://discord.com/invite/eN6D2HQ4aX"
							);
							script.async = true;
							document.head.appendChild(script);
						});
					`,
        },
      ],
      social: [
        {
          icon: "twitter",
          label: "Twitter",
          href: "https://x.com/llama_index",
        },
        {
          icon: "linkedin",
          label: "LinkedIn",
          href: "https://www.linkedin.com/company/llamaindex",
        },
        {
          icon: "blueSky",
          label: "Bluesky",
          href: "https://bsky.app/profile/llamaindex.bsky.social",
        },
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/run-llama/llama_index/",
        },
      ],
      logo: {
        light: "./src/assets/llamaindex-dark.svg",
        dark: "./src/assets/llamaindex-light.svg",
        replacesTitle: true,
      },
      favicon: "/logo-dark.png",
      components: {
        SiteTitle: "./src/components/SiteTitle.astro",
        Header: "./src/components/Header.astro",
      },
      sidebar: [
        {
          label: "LlamaIndex Framework",
          autogenerate: { directory: "framework", collapsed: true },
        },
        {
          label: "Examples",
          items: [
            {
              label: "LLMs",
              collapsed: true,
              autogenerate: { directory: "framework/examples/llm" },
            },
            {
              label: "Embeddings",
              collapsed: true,
              autogenerate: { directory: "framework/examples/embeddings" },
            },
            {
              label: "Vector Stores",
              collapsed: true,
              autogenerate: { directory: "framework/examples/vector_stores" },
            },
            {
              label: "Retrievers",
              collapsed: true,
              autogenerate: { directory: "framework/examples/retrievers" },
            },
          ],
        },
      ],
    }),
    AutoImport({
      imports: [
        {
          "@icons-pack/react-simple-icons": [
            "SiBun",
            "SiCloudflareworkers",
            "SiDeno",
            "SiNodedotjs",
            "SiTypescript",
            "SiVite",
            "SiNextdotjs",
            "SiDiscord",
            "SiGithub",
            "SiNpm",
            "SiX",
          ],
          "@astrojs/starlight/components": [
            "Card",
            "CardGrid",
            "LinkCard",
            "Icon",
            "Tabs",
            "TabItem",
            "Aside",
          ],
        },
      ],
    }),
    react(),
  ],
  markdown: {
    remarkPlugins: [findBrokenLinks],
  },
  image: {
    service: passthroughImageService(),
  },
});

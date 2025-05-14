document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.defer = true;
    script.type = "module";
    script.id = "runllm-widget-script";
    script.src =
      "https://widget.runllm.com";
    script.setAttribute("runllm-preset", "mkdocs");
    script.setAttribute("runllm-server-address", "https://api.runllm.com");
    script.setAttribute("runllm-assistant-id", "209");
    script.setAttribute("runllm-position", "BOTTOM_RIGHT");
    script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
    script.setAttribute("runllm-disable-ask-a-person", true);
    script.setAttribute(
      "runllm-slack-community-url",
      "https://discord.com/invite/eN6D2HQ4aX"
    );
    script.setAttribute("runllm-name", "RunLLM");

    document.head.appendChild(script);
  });

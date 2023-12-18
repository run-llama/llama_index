function killTheBug() {
  let elements = document.querySelectorAll("readthedocs-hotkeys");
  if (elements.length) {
    for (el of elements) {
      el.docDiffHotKeyEnabled = false;
    }
  }
}
// try to remove the 'd' handler every seconds
setInterval(killTheBug, 1000);

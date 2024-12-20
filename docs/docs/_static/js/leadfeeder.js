(function (ss, ex) {
  window.ldfdr =
    window.ldfdr ||
    function () {
      (ldfdr._q = ldfdr._q || []).push([].slice.call(arguments));
    };
  (function (d, s) {
    fs = d.getElementsByTagName(s)[0];
    function ce(src) {
      var cs = d.createElement(s);
      cs.src = src;
      cs.async = 1;
      fs.parentNode.insertBefore(cs, fs);
    }
    ce(
      "https://sc.lfeeder.com/lftracker_v1_" +
        ss +
        (ex ? "_" + ex : "") +
        ".js",
    );
  })(document, "script");
})("Xbp1oaEnqwn8EdVj");

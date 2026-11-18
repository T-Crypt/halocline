(function () {
  "use strict";

  var gauge = document.querySelector(".gauge");
  if (gauge) {
    var needle = gauge.querySelector(".gauge-needle");
    var readout = gauge.querySelector(".gauge-readout");
    var top = Number(gauge.getAttribute("data-top")) || 1;
    var layers = Array.prototype.slice.call(document.querySelectorAll("[data-depth]"));
    var ticking = false;

    var place = function () {
      ticking = false;
      var mark = window.scrollY + window.innerHeight * 0.35;
      var current = layers[0];
      var next = null;
      for (var i = 0; i < layers.length; i += 1) {
        if (layers[i].offsetTop <= mark) {
          current = layers[i];
          next = layers[i + 1] || null;
        }
      }
      var start = current.offsetTop;
      var end = next ? next.offsetTop : document.documentElement.scrollHeight;
      var span = Math.max(1, end - start);
      var t = Math.min(1, Math.max(0, (mark - start) / span));
      var from = Number(current.getAttribute("data-depth")) || 0;
      var to = next ? Number(next.getAttribute("data-depth")) || from : from;
      var depth = Math.round(from + (to - from) * t);
      var frac = Math.min(1, Math.max(0, depth / top));
      needle.style.top = (frac * 100).toFixed(3) + "%";
      readout.textContent = depth.toLocaleString("en-US") + " tokens";
    };

    var request = function () {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(place);
      }
    };

    window.addEventListener("scroll", request, { passive: true });
    window.addEventListener("resize", request);
    place();
  }

  var tip = document.querySelector(".chart-tip");
  if (tip) {
    var hide = function () {
      tip.classList.remove("on");
      tip.textContent = "";
    };
    var show = function (target, x, y) {
      var label = target.getAttribute("data-label");
      if (!label) {
        return;
      }
      tip.textContent = label;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
      tip.classList.add("on");
    };

    Array.prototype.forEach.call(document.querySelectorAll(".chart .hover"), function (node) {
      node.addEventListener("pointerenter", function (event) {
        show(node, event.clientX, event.clientY);
      });
      node.addEventListener("pointermove", function (event) {
        if (tip.classList.contains("on")) {
          tip.style.left = event.clientX + "px";
          tip.style.top = event.clientY + "px";
        }
      });
      node.addEventListener("pointerleave", hide);
      node.addEventListener("focus", function () {
        var box = node.getBoundingClientRect();
        show(node, box.left + box.width / 2, box.top);
      });
      node.addEventListener("blur", hide);
    });
  }
})();

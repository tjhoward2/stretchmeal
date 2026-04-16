/**
 * Zone Tagger — draggable guide lines for rug border annotation.
 *
 * Renders four guide lines (top, bottom, left, right) over a design
 * thumbnail. Users drag lines to correct border positions, then save.
 */

(function () {
  "use strict";

  let currentDesign = null;
  let guides = {};    // { top, bottom, left, right } → DOM elements
  let imgEl = null;
  let containerEl = null;

  // Scale factor: master pixels → display pixels
  let scaleX = 1;
  let scaleY = 1;

  // Current border values in master pixels
  let borders = { top: 0, bottom: 0, left: 0, right: 0 };

  // ── Design list ──────────────────────────────────────────────

  async function loadDesigns() {
    const resp = await fetch("/api/designs");
    const designs = await resp.json();

    const listEl = document.getElementById("design-list");
    const countEl = document.getElementById("review-count");
    var reviewCount = designs.filter(function (d) { return d.zone_map.needs_review; }).length;
    countEl.textContent = designs.length;

    listEl.innerHTML = "";
    if (designs.length === 0) {
      listEl.innerHTML = '<div style="padding:16px;color:#aaa;">No designs found</div>';
      return;
    }

    designs.forEach(function (d) {
      var needsReview = d.zone_map.needs_review;
      const item = document.createElement("div");
      item.className = "design-item";
      item.dataset.id = d.design_id;
      item.innerHTML =
        '<div class="id">Design ' + d.design_id + "</div>" +
        '<div class="meta">Confidence: ' + (d.zone_map.confidence * 100).toFixed(0) + "%</div>" +
        (needsReview ? '<div class="review-tag">Needs Review</div>' : '<div class="meta" style="color:#27ae60;">OK</div>');
      item.addEventListener("click", function () {
        document.querySelectorAll(".design-item").forEach(function (el) { el.classList.remove("active"); });
        item.classList.add("active");
        openDesign(d.design_id);
      });
      listEl.appendChild(item);
    });
  }

  // ── Editor ───────────────────────────────────────────────────

  async function openDesign(designId) {
    currentDesign = designId;

    const resp = await fetch("/api/designs/" + designId + "/zones");
    const zone = await resp.json();

    borders.top = zone.border_top_px || 0;
    borders.bottom = zone.border_bottom_px || 0;
    borders.left = zone.border_left_px || 0;
    borders.right = zone.border_right_px || 0;

    var masterW = zone.master_width_px;
    var masterH = zone.master_height_px;

    var editor = document.getElementById("editor");
    editor.innerHTML =
      '<div class="image-container" id="img-container">' +
        '<img id="design-img" src="/api/designs/' + designId + '/thumbnail" draggable="false">' +
        '<div class="guide horizontal" id="guide-top"></div>' +
        '<div class="guide horizontal" id="guide-bottom"></div>' +
        '<div class="guide vertical" id="guide-left"></div>' +
        '<div class="guide vertical" id="guide-right"></div>' +
        '<div class="guide-label" id="label-top"></div>' +
        '<div class="guide-label" id="label-bottom"></div>' +
        '<div class="guide-label" id="label-left"></div>' +
        '<div class="guide-label" id="label-right"></div>' +
      "</div>" +
      '<div class="controls">' +
        '<label>Top <input type="number" id="input-top" min="0"></label>' +
        '<label>Bottom <input type="number" id="input-bottom" min="0"></label>' +
        '<label>Left <input type="number" id="input-left" min="0"></label>' +
        '<label>Right <input type="number" id="input-right" min="0"></label>' +
        '<button class="btn-save" id="btn-save">Save</button>' +
        '<button class="btn-reset" id="btn-reset">Reset</button>' +
        '<button class="btn-delete" id="btn-delete">Delete</button>' +
        '<span class="status" id="status">Saved!</span>' +
      "</div>";

    imgEl = document.getElementById("design-img");
    containerEl = document.getElementById("img-container");

    guides.top = document.getElementById("guide-top");
    guides.bottom = document.getElementById("guide-bottom");
    guides.left = document.getElementById("guide-left");
    guides.right = document.getElementById("guide-right");

    imgEl.addEventListener("load", function () {
      scaleX = imgEl.naturalWidth / masterW;
      scaleY = imgEl.naturalHeight / masterH;
      positionGuides();
      updateInputs();
    });

    // If image already cached
    if (imgEl.complete) {
      scaleX = imgEl.naturalWidth / masterW;
      scaleY = imgEl.naturalHeight / masterH;
      positionGuides();
      updateInputs();
    }

    // Dragging
    makeDraggable("top", "horizontal");
    makeDraggable("bottom", "horizontal");
    makeDraggable("left", "vertical");
    makeDraggable("right", "vertical");

    // Input fields
    ["top", "bottom", "left", "right"].forEach(function (side) {
      document.getElementById("input-" + side).addEventListener("change", function () {
        borders[side] = parseInt(this.value) || 0;
        positionGuides();
      });
    });

    // Save
    document.getElementById("btn-save").addEventListener("click", saveZones);
    document.getElementById("btn-reset").addEventListener("click", function () {
      openDesign(designId);
    });
    document.getElementById("btn-delete").addEventListener("click", function () {
      deleteDesign(designId);
    });
  }

  function positionGuides() {
    if (!imgEl || !imgEl.complete) return;

    var displayW = imgEl.clientWidth;
    var displayH = imgEl.clientHeight;

    var topPx = borders.top * scaleY;
    var bottomPx = borders.bottom * scaleY;
    var leftPx = borders.left * scaleX;
    var rightPx = borders.right * scaleX;

    // Convert from master-scaled to display coordinates
    var topDisplay = (topPx / imgEl.naturalHeight) * displayH;
    var bottomDisplay = displayH - (bottomPx / imgEl.naturalHeight) * displayH;
    var leftDisplay = (leftPx / imgEl.naturalWidth) * displayW;
    var rightDisplay = displayW - (rightPx / imgEl.naturalWidth) * displayW;

    guides.top.style.top = topDisplay + "px";
    guides.bottom.style.top = bottomDisplay + "px";
    guides.left.style.left = leftDisplay + "px";
    guides.right.style.left = rightDisplay + "px";

    // Labels
    var lTop = document.getElementById("label-top");
    lTop.textContent = "top: " + borders.top + "px";
    lTop.style.top = (topDisplay + 4) + "px";
    lTop.style.left = "4px";

    var lBottom = document.getElementById("label-bottom");
    lBottom.textContent = "bottom: " + borders.bottom + "px";
    lBottom.style.top = (bottomDisplay - 18) + "px";
    lBottom.style.left = "4px";

    var lLeft = document.getElementById("label-left");
    lLeft.textContent = "left: " + borders.left + "px";
    lLeft.style.left = (leftDisplay + 4) + "px";
    lLeft.style.top = "4px";

    var lRight = document.getElementById("label-right");
    lRight.textContent = "right: " + borders.right + "px";
    lRight.style.left = (rightDisplay - 80) + "px";
    lRight.style.top = "4px";

    updateInputs();
  }

  function updateInputs() {
    ["top", "bottom", "left", "right"].forEach(function (side) {
      var el = document.getElementById("input-" + side);
      if (el) el.value = borders[side];
    });
  }

  // ── Drag logic ───────────────────────────────────────────────

  function makeDraggable(side, orientation) {
    var guideEl = guides[side];

    guideEl.addEventListener("mousedown", function (e) {
      e.preventDefault();
      var rect = containerEl.getBoundingClientRect();
      var displayW = imgEl.clientWidth;
      var displayH = imgEl.clientHeight;

      function onMove(ev) {
        if (orientation === "horizontal") {
          var y = ev.clientY - rect.top;
          y = Math.max(0, Math.min(displayH, y));

          if (side === "top") {
            borders.top = Math.round((y / displayH) * imgEl.naturalHeight / scaleY);
          } else {
            borders.bottom = Math.round(((displayH - y) / displayH) * imgEl.naturalHeight / scaleY);
          }
        } else {
          var x = ev.clientX - rect.left;
          x = Math.max(0, Math.min(displayW, x));

          if (side === "left") {
            borders.left = Math.round((x / displayW) * imgEl.naturalWidth / scaleX);
          } else {
            borders.right = Math.round(((displayW - x) / displayW) * imgEl.naturalWidth / scaleX);
          }
        }
        positionGuides();
      }

      function onUp() {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      }

      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  // ── Save ─────────────────────────────────────────────────────

  async function saveZones() {
    if (!currentDesign) return;

    var body = {
      border_top_px: borders.top,
      border_bottom_px: borders.bottom,
      border_left_px: borders.left,
      border_right_px: borders.right,
    };

    var resp = await fetch("/api/designs/" + currentDesign + "/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (resp.ok) {
      var statusEl = document.getElementById("status");
      statusEl.classList.add("show");
      setTimeout(function () { statusEl.classList.remove("show"); }, 2000);

      // Refresh the list (design may no longer need review)
      loadDesigns();
    }
  }

  // ── Delete ───────────────────────────────────────────────────

  async function deleteDesign(designId) {
    if (!confirm("Delete design " + designId + "? This cannot be undone.")) return;

    var resp = await fetch("/api/designs/" + designId, {
      method: "DELETE",
    });

    if (resp.ok) {
      currentDesign = null;
      document.getElementById("editor").innerHTML =
        '<div class="editor-empty">Design deleted. Select another design.</div>';
      loadDesigns();
    }
  }

  // ── Init ─────────────────────────────────────────────────────

  loadDesigns();
})();

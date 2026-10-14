// SentryAML 复核界面（R11b）的全部脚本。
// 页面上的每个按钮都调用已经测好的 JSON 接口（R11a / R11b），不另设表单专用的写入路径。
// 规则：所有文字用 textContent 写入，从不使用 innerHTML——数据库或模型给的内容只当作文字。
(function () {
    "use strict";
    var csrfMeta = document.querySelector('meta[name="csrf"]');
    var csrf = csrfMeta ? csrfMeta.getAttribute("content") : "";
  
    function showMessage(text, ok) {
      var box = document.getElementById("message");
      if (!box) { return; }
      box.textContent = text;
      box.className = ok ? "message ok" : "message";
      box.hidden = false;
    }
  
    function errorText(status, data) {
      var d = data && data.detail;
      if (d && typeof d === "object" && !Array.isArray(d)) {
        return (d.hint || d.message || d.code || "请求被拒绝") + "（" + status + "）";
      }
      if (Array.isArray(d)) {           // 请求格式校验失败（422）：字段缺失、多余字段、取值不对
        return d.map(function (x) {
          var where = (x.loc || []).slice(1).join(".");          // 整个请求体的校验（model_validator）没有字段名
          var msg = String(x.msg || "").replace(/^Value error, /, "");
          return where ? where + "：" + msg : msg;
        }).join("；") + "（" + status + "）";
      }
      return "请求失败（" + status + "）";
    }
  
    async function post(url, body) {
      var r = await fetch(url, {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify(body || {})
      });
      var data = null;
      try { data = await r.json(); } catch (e) { data = null; }
      return { ok: r.ok, status: r.status, data: data };
    }
  
    // 按钮的请求内容 = data-body（固定字段：动作、页面加载时看到的状态、草稿版本）+ data-fields 列出的输入框
    function collect(btn) {
      var body = btn.dataset.body ? JSON.parse(btn.dataset.body) : {};
      (btn.dataset.fields || "").split(",").filter(Boolean).forEach(function (f) {
        if (f === "drop_categories") {
          body.drop_categories = Array.prototype.map.call(
            document.querySelectorAll('input[name="drop"]:checked'), function (x) { return x.value; });
          return;
        }
        var el = document.getElementById(f);
        if (!el) { return; }
        var v = el.value.trim();
        if (f === "narrative") {          // 理由没改过就不发送（只删类别）
          if (v !== (el.dataset.original || "").trim()) { body.narrative = v; }
          return;
        }
        if (v) { body[f] = v; }
      });
      return body;
    }
  
    document.addEventListener("click", async function (ev) {
      var btn = ev.target.closest("[data-post]");
      if (btn) {
        ev.preventDefault();
        var label = btn.textContent;
        btn.disabled = true;
        if (btn.dataset.busy) { btn.textContent = btn.dataset.busy; }
        var res = await post(btn.dataset.post, collect(btn));
        if (res.ok) { window.location.reload(); return; }
        btn.disabled = false;
        btn.textContent = label;
        showMessage(errorText(res.status, res.data), false);
        return;
      }
      if (ev.target.closest("[data-logout]")) {
        await post("/api/logout", {});
        window.location.href = "/login";
      }
    });
  
    var form = document.getElementById("login-form");
    if (form) {
      form.addEventListener("submit", async function (ev) {
        ev.preventDefault();
        var fd = new FormData(form);
        var r = await fetch("/api/login", {
          method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reviewer_id: fd.get("reviewer_id"), password: fd.get("password") })
        });
        if (r.ok) { window.location.href = "/queue"; return; }
        var data = null;
        try { data = await r.json(); } catch (e) { data = null; }
        showMessage(errorText(r.status, data), false);
      });
    }
  })();